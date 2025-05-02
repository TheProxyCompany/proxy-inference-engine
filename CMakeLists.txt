cmake_minimum_required(VERSION 3.20)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)
set(CMAKE_CXX_STANDARD 23)

project(proxy_inference_engine LANGUAGES CXX)
option(BUILD_TESTING "Build the tests" OFF)
# add C++ source directories
add_subdirectory(src/pie_core)
# add C++ tests
if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests/cpp)
endif()
