# TRACCC library, part of the ACTS project (R&D line)
#
# (c) 2021-2025 CERN for the benefit of the ACTS project
#
# Mozilla Public License Version 2.0

# Project include(s).
include( traccc-compiler-options-cpp )

# Set up a common library for the examples.
add_library( traccc_examples_common STATIC
   "common/seq_example.hpp"
   "common/seq_example.cpp"
   "common/throughput_mt.hpp"
   "common/throughput_mt.ipp"
   "common/throughput_st.hpp"
   "common/throughput_st.ipp" )
target_link_libraries( traccc_examples_common
    PUBLIC
    traccc::core
    traccc::options
    INTERFACE
    vecmem::core
    traccc::io
    traccc::performance )

# Add all the subdirectories that can be built.
add_subdirectory(cpu)

if (TRACCC_BUILD_CUDA)
  add_subdirectory(cuda)
endif()

if (TRACCC_BUILD_SYCL)
  add_subdirectory(sycl)
endif()

if (TRACCC_BUILD_KOKKOS)
  add_subdirectory(kokkos)
endif()

if (TRACCC_BUILD_ALPAKA)
  add_subdirectory(alpaka)
endif()
