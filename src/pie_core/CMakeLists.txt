cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0087 NEW)

set(PROJECT_NAME proxy_inference_engine) # Top-level project name
set(EXTENSION_NAME pie_core)             # Python module name
set(CORE_LIB_NAME pie_core_lib)          # C++ Static Library name
set(ENGINE_EXE_NAME pie_engine)          # C++ Executable name

# --- C++ Settings & Build Type ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 \
        -fno-omit-frame-pointer \
        -Wall -Wextra \
        -Wformat=2 -Wformat-security \
        -Wcast-align -Wcast-qual \
        -Wdouble-promotion \
        -Wmissing-include-dirs \
        -Wnull-dereference -Wredundant-decls \
        -Wshadow -Wstrict-aliasing=2 \
        -Wuninitialized \
        -D_GLIBCXX_DEBUG \
        -D_GLIBCXX_DEBUG_PEDANTIC \
        -D_FORTIFY_SOURCE=2"
    )
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 \
        -mtune=native \
        -flto=auto \
        -DNDEBUG \
        -ffunction-sections \
        -fdata-sections \
        -ftree-vectorize \
        -fstack-protector-strong \
        -D_FORTIFY_SOURCE=2 \
        -funroll-loops \
        -finline-functions"
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
            -fmerge-all-constants \
            -fvectorize \
            -fslp-vectorize")
    endif()
endif()

if(NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
endif()

# Build type configuration
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "Coverage")
endif()

# --- Python Configuration ---
set(Python_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
message(STATUS "Using Python site-packages (Python_SITELIB): ${Python_SITELIB}")

# --- Find External Dependencies ---
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind EXCLUDE_FROM_ALL)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/mlx EXCLUDE_FROM_ALL)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog EXCLUDE_FROM_ALL)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/tokenizers EXCLUDE_FROM_ALL)

find_package(nlohmann_json 3.12.0 REQUIRED)
find_package(Boost)
if(Boost_FOUND)
    message(STATUS "Boost found. Version: ${Boost_VERSION_STRING}")
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
else()
    message(STATUS "Boost not found via find_package. Attempting FetchContent for Boost headers.")
    include(FetchContent)
    FetchContent_Declare(
        boost_headers
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.88.0
    )
    FetchContent_GetProperties(boost_headers)
    if(NOT boost_headers_POPULATED)
        FetchContent_Populate(boost_headers)
    endif()
    include_directories(SYSTEM ${boost_headers_SOURCE_DIR})
    message(STATUS "Using Boost headers from FetchContent directory: ${boost_headers_SOURCE_DIR}")
endif()

# --- Find installed pse_core package location ---
set(pse_core_DIR "${Python_SITELIB}/pse_core/sdk/cmake")
message(STATUS "Setting pse_core_DIR to: ${pse_core_DIR}")
find_package(pse_core REQUIRED)

# --- Check results and proceed ---
# Check if the specific library target 'pse_core::pse_sdk' exists after find_package
if(TARGET pse_core::pse_sdk)
    message(STATUS "Found pse_core target: pse_core::pse_sdk")
    # Set the variable to the correct target name
    set(PSE_CORE_LINK_TARGET pse_core::pse_sdk)
else()
    # If the target doesn't exist, check for the older variable style as a fallback (less likely)
    if(pse_core_LIBRARIES)
        message(STATUS "Found pse_core libraries variable: ${pse_core_LIBRARIES}")
        set(PSE_CORE_LINK_TARGET ${pse_core_LIBRARIES})
    endif()
endif()

find_package(MLX CONFIG REQUIRED)
target_compile_options(mlx PRIVATE -w)

# --- Global Include Directories ---
include_directories(
    SYSTEM
    ${pse_core_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/nanobind/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/mlx/mlx
    ${CMAKE_CURRENT_SOURCE_DIR}/external/tokenizers/include
)

# --- Source Files, Module Definition, Target Properties ---
file(GLOB_RECURSE PIE_CORE_LIB_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/layers/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/logit_processors/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/models/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/models/**/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/samplers/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sequence/*.cpp"
)

# Define source for Python bindings
set(PIE_BINDINGS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp")

# Define source for the standalone C++ engine executable
set(PIE_ENGINE_MAIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

add_library(${CORE_LIB_NAME} STATIC ${PIE_CORE_LIB_SRC})

target_include_directories(${CORE_LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(${CORE_LIB_NAME} PUBLIC
    ${PSE_CORE_LINK_TARGET}
    mlx
    tokenizers_cpp
    nlohmann_json::nlohmann_json
    spdlog::spdlog_header_only
)

target_compile_options(${CORE_LIB_NAME} PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;-fcoverage-mapping>
)
target_link_options(${CORE_LIB_NAME} PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;-fcoverage-mapping>
)

# --- Build Metal Library ---
if(MLX_BUILD_METAL)
    message(STATUS "Configuring Metal library build for pie_core")
    mlx_build_metallib(
        TARGET pie_core_metallib
        TITLE ${EXTENSION_NAME}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/paged_attention.metal
        INCLUDE_DIRS
             ${CMAKE_CURRENT_SOURCE_DIR}/include
             ${MLX_INCLUDE_DIRS}
        OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_dependencies(${CORE_LIB_NAME} pie_core_metallib)
endif()

nanobind_add_module(
    ${EXTENSION_NAME}
    STABLE_ABI NB_STATIC LTO NOMINSIZE
    ${PIE_BINDINGS_SRC}
)

target_link_libraries(
    ${EXTENSION_NAME}
    PRIVATE
        ${CORE_LIB_NAME}
)

target_compile_options(${EXTENSION_NAME} PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;-fcoverage-mapping>
)
target_link_options(${EXTENSION_NAME} PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;-fcoverage-mapping>
)

target_compile_definitions(
    ${EXTENSION_NAME}
    PRIVATE
        MODULE_NAME=$<TARGET_FILE_BASE_NAME:${EXTENSION_NAME}>
        VERSION_INFO="${PY_FULL_VERSION}"
        $<$<CONFIG:Debug>:ENABLE_ASSERTIONS>
)

# Install Metal library alongside Python module
if(MLX_BUILD_METAL AND TARGET pie_core_metallib)
    install(
        FILES ${CMAKE_CURRENT_BINARY_DIR}/pie_core.metallib
        DESTINATION ${PROJECT_NAME}
        COMPONENT python_modules
    )
endif()

add_executable(${ENGINE_EXE_NAME} ${PIE_ENGINE_MAIN_SRC})
target_link_libraries(${ENGINE_EXE_NAME} PRIVATE ${CORE_LIB_NAME})

target_compile_options(${ENGINE_EXE_NAME} PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;-fcoverage-mapping>
)
target_link_options(${ENGINE_EXE_NAME} PRIVATE
    $<$<CONFIG:Coverage>:-fprofile-instr-generate;-fcoverage-mapping>
)

# --- Installation ---
install(
    TARGETS ${EXTENSION_NAME}
    LIBRARY DESTINATION ${PROJECT_NAME}
    COMPONENT python_modules
)

install(
    TARGETS ${ENGINE_EXE_NAME}
    RUNTIME DESTINATION bin
    COMPONENT engine_executable
)
