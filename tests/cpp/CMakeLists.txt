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
foreach(src ${test_files})
    get_filename_component(target_name ${src} NAME_WE)
    add_executable(${target_name} ${src})
    target_compile_options(${target_name} PRIVATE "-fno-stack-protector")
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)
    target_link_libraries(${target_name}
        PRIVATE pie_core_lib
                GTest::gtest_main
    )
    gtest_discover_tests(${target_name})
endforeach()
