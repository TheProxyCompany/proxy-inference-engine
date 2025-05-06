cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PIE_USE_TRACY "Enable Tracy profiler" OFF)

# --- Fetch Google Benchmark ---
include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG main
  GIT_SHALLOW TRUE
)
# Disable benchmark tests and install
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)
# --- Fetch Tracy profiler ---
if(PIE_USE_TRACY)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(tracy)

    # Create Tracy library - using STATIC instead of INTERFACE
    add_library(tracy_client STATIC ${tracy_SOURCE_DIR}/public/TracyClient.cpp)
    target_include_directories(tracy_client PUBLIC ${tracy_SOURCE_DIR}/public)
    target_compile_definitions(tracy_client PUBLIC TRACY_ENABLE)
    message(STATUS "Tracy profiler enabled at ${tracy_SOURCE_DIR}")
endif()

# Define benchmark executable
add_executable(pie_benchmarks
    core/page_allocator_benchmark.cpp
)

# Add include directories
target_include_directories(pie_benchmarks PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/benchmarks/utils
    ${PROJECT_SOURCE_DIR}/src/pie_core/include
)

# Link dependencies - using only benchmark_main to avoid duplicate linking
target_link_libraries(pie_benchmarks PRIVATE
    pie_core_lib
    benchmark::benchmark_main
)

if(PIE_USE_TRACY)
    target_link_libraries(pie_benchmarks PRIVATE tracy_client)
endif()

# Set compile options for benchmarks
target_compile_options(pie_benchmarks PRIVATE
    $<$<NOT:$<CONFIG:Coverage>>:-O3>
    $<$<NOT:$<CONFIG:Coverage>>:-DNDEBUG>
    $<$<NOT:$<CONFIG:Coverage>>:-march=native>
    $<$<NOT:$<CONFIG:Coverage>>:-mtune=native>
    -Wall
    -Wextra
    -Wno-unused-parameter
)

target_compile_options(pie_benchmarks PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;>
)
target_link_options(pie_benchmarks PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;>
)

# Include platform-specific optimization flags
if(APPLE)
    # macOS-specific optimization flags
    target_compile_options(pie_benchmarks PRIVATE $<$<NOT:$<CONFIG:Coverage>>:-mcpu=apple-m1>)
elseif(UNIX AND NOT APPLE)
    # Linux-specific optimization flags if needed
    target_compile_options(pie_benchmarks PRIVATE
        $<$<NOT:$<CONFIG:Coverage>>:-mavx2>
        $<$<NOT:$<CONFIG:Coverage>>:-mfma>
        $<$<NOT:$<CONFIG:Coverage>>:-msse4.2>
    )
endif()

# Install benchmark executable
install(
    TARGETS pie_benchmarks
    RUNTIME DESTINATION bin
    COMPONENT benchmarks
)

add_custom_target(run_benchmarks
    COMMAND $<TARGET_FILE:pie_benchmarks>
    DEPENDS pie_benchmarks
    COMMENT "Running benchmarks..."
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
