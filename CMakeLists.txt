cmake_minimum_required(VERSION 3.20)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

project(proxy_inference_engine LANGUAGES CXX)
add_subdirectory(src/pie_core)
