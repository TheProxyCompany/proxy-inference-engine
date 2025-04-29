cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0087 NEW)

set(EXTENSION_NAME pie_core)

project(${EXTENSION_NAME} LANGUAGES CXX)

# C++ settings
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 \
        -fsanitize=undefined \
        -fno-omit-frame-pointer \
        -Wall -Wextra -Wpedantic \
        -Wformat=2 -Wformat-security \
        -Wcast-align -Wcast-qual \
        -Wconversion -Wdouble-promotion \
        -Wmissing-include-dirs \
        -Wnull-dereference -Wredundant-decls \
        -Wshadow -Wstrict-aliasing=2 \
        -Wuninitialized \
        -D_GLIBCXX_DEBUG \
        -D_GLIBCXX_DEBUG_PEDANTIC \
        -D_FORTIFY_SOURCE=2"
    )
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
        -Ofast \
        -march=native \
        -mtune=native \
        -flto=auto \
        -DNDEBUG \
        -ffunction-sections \
        -fdata-sections \
        -ftree-vectorize \
        -fstack-protector-strong \
        -D_FORTIFY_SOURCE=2 \
        -funroll-loops \
        -finline-functions \
        -fno-plt"
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
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

# Python configuration
set(DEV_MODULE Development.Module)

set(Python_VIRTUALENV FIRST)
find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# External dependencies
add_subdirectory(${CMAKE_SOURCE_DIR}/external/nanobind)

# Include directories
include_directories(
    SYSTEM
    ${CMAKE_SOURCE_DIR}/external/nanobind/include
    ${CMAKE_SOURCE_DIR}/external/mlx/include
    )
include_directories( ${CMAKE_SOURCE_DIR}/include )

# Source files
file(GLOB_RECURSE pie_core_src CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# Main module configuration
nanobind_add_module(
    ${EXTENSION_NAME}
    STABLE_ABI LTO NOMINSIZE
    ${pie_core_src}
)

target_compile_definitions(
    ${EXTENSION_NAME}
    PRIVATE
        MODULE_NAME=$<TARGET_FILE_BASE_NAME:${EXTENSION_NAME}>
        VERSION_INFO="${PY_FULL_VERSION}"
        $<$<CONFIG:Debug>:ENABLE_ASSERTIONS>
)

target_include_directories(
    ${EXTENSION_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
)

# Installation
install(
    TARGETS ${EXTENSION_NAME}
    LIBRARY DESTINATION .
    COMPONENT python_modules
)
