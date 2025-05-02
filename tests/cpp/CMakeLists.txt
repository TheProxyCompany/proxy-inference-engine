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

set(Tests_Target_Name "pie_core_tests")
add_executable(${Tests_Target_Name} ${test_files})
target_compile_options(
  ${Tests_Target_Name} PRIVATE
    "-fno-stack-protector"
)
target_compile_options(
  ${Tests_Target_Name} PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;-fcoverage-mapping>
)
target_link_options(
  ${Tests_Target_Name} PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;-fcoverage-mapping>
)
set_target_properties(
  ${Tests_Target_Name}
  PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION $<$<NOT:$<CONFIG:Coverage>>:OFF>
)
target_link_libraries(
  ${Tests_Target_Name}
  PRIVATE
    pie_core_lib
    GTest::gtest_main
)
gtest_discover_tests(${Tests_Target_Name})
