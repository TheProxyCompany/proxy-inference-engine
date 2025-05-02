cmake_minimum_required(VERSION 3.20)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

project(proxy_inference_engine LANGUAGES CXX)

# add C++ source directories
add_subdirectory(src/pie_core)

option(BUILD_TESTING "Build the tests" ON)
# add C++ tests
if(BUILD_TESTING)
    include(CTest)
    add_subdirectory(tests/cpp)
endif()
