cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)

project(proxy_inference_engine LANGUAGES CXX)

find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

if(APPLE)
  set(CMAKE_BUILD_PARALLEL_LEVEL $(sysctl -n hw.physicalcpu))
elseif(UNIX)
  set(CMAKE_BUILD_PARALLEL_LEVEL $(nproc))
elseif(WIN32)
  set(CMAKE_BUILD_PARALLEL_LEVEL $ENV{NUMBER_OF_PROCESSORS})
else()
  set(CMAKE_BUILD_PARALLEL_LEVEL 4)
endif()

# add C++ source directories
add_subdirectory(src/pie_core)

option(PIE_BUILD_TESTING "Build the tests" OFF)
option(PIE_BUILD_BENCHMARKS "Build benchmarks" OFF)

# add C++ tests
if(PIE_BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests/cpp)
endif()

# add benchmarks
if(PIE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks/cpp)
endif()
