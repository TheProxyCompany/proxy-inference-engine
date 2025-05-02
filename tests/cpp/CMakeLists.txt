cmake_minimum_required(VERSION 3.20)
project(pie_tests LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# glob files in tests/cpp/
file(GLOB_RECURSE test_files "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(src ${test_sources})
    get_filename_component(tgt ${src} NAME_WE)
    add_executable(${tgt} ${src})
    target_link_libraries(${tgt}
        PRIVATE pie_core_lib
                GTest::gtest_main
    )
    gtest_discover_tests(${tgt})
endforeach()
