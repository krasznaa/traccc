# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cuda )

# External(s).
find_package( CUDAToolkit REQUIRED )

traccc_add_executable( seq_example_cuda "seq_example_cuda.cpp"
   LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options detray::detectors detray::io
                  traccc_examples_common )
traccc_add_executable( seeding_example_cuda "seeding_example_cuda.cpp"
   LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options traccc_examples_common )
traccc_add_executable( truth_finding_example_cuda "truth_finding_example_cuda.cpp"
   LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options traccc_examples_common )
traccc_add_executable( truth_fitting_example_cuda "truth_fitting_example_cuda.cpp"
   LINK_LIBRARIES vecmem::core vecmem::cuda traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options traccc_examples_common )
#
# Set up the "throughput applications".
#
add_library( traccc_examples_cuda STATIC
   "full_chain_algorithm.hpp"
   "full_chain_algorithm.cpp" )
target_link_libraries( traccc_examples_cuda
   PUBLIC CUDA::cudart vecmem::core vecmem::cuda detray::core detray::detectors
          traccc::core traccc::device_common traccc::cuda
          traccc_examples_common )

traccc_add_executable( throughput_st_cuda "throughput_st.cpp"
   LINK_LIBRARIES indicators::indicators vecmem::core vecmem::cuda
                  detray::detectors detray::io traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options traccc_examples_cuda )

traccc_add_executable( throughput_mt_cuda "throughput_mt.cpp"
   LINK_LIBRARIES indicators::indicators TBB::tbb vecmem::core vecmem::cuda
                  detray::detectors detray::io traccc::io traccc::performance
                  traccc::core traccc::device_common traccc::cuda
                  traccc::options traccc_examples_cuda )
