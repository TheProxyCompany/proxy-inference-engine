#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# --- Configuration ---
BUILD_DIR="build"
CORE_LIB_NAME="pie_core_lib"
ENGINE_EXE_NAME="pie_engine"

# --- Paths ---
PROFRAW_PATTERN="${BUILD_DIR}/tests/cpp/*.profraw"
PROFDATA_FILE="${BUILD_DIR}/coverage.profdata"
# Define path to the test executable
TEST_EXE_PATH="${BUILD_DIR}/tests/cpp/test_page_allocator" # Path to test executable

# --- Find LLVM Tools ---
LLVM_PROFDATA_CMD=""
LLVM_COV_CMD=""

# Check standard PATH first
if command -v llvm-profdata &> /dev/null; then
    LLVM_PROFDATA_CMD=$(command -v llvm-profdata)
fi
if command -v llvm-cov &> /dev/null; then
    LLVM_COV_CMD=$(command -v llvm-cov)
fi

# If not found in PATH, check common Homebrew LLVM paths
if [ -z "$LLVM_PROFDATA_CMD" ] || [ -z "$LLVM_COV_CMD" ]; then
    # Common Homebrew LLVM path (might vary based on version)
    BREW_LLVM_PREFIX=""
    if command -v brew &> /dev/null; then
        BREW_LLVM_PREFIX=$(brew --prefix llvm 2>/dev/null || true)
    fi

    if [ -n "$BREW_LLVM_PREFIX" ] && [ -d "$BREW_LLVM_PREFIX/bin" ]; then
        if [ -z "$LLVM_PROFDATA_CMD" ] && [ -x "$BREW_LLVM_PREFIX/bin/llvm-profdata" ]; then
            LLVM_PROFDATA_CMD="$BREW_LLVM_PREFIX/bin/llvm-profdata"
        fi
        if [ -z "$LLVM_COV_CMD" ] && [ -x "$BREW_LLVM_PREFIX/bin/llvm-cov" ]; then
            LLVM_COV_CMD="$BREW_LLVM_PREFIX/bin/llvm-cov"
        fi
    fi
fi

# Final check - if variables are still empty, exit with error
if [ -z "$LLVM_PROFDATA_CMD" ]; then
    echo "Error: Could not automatically locate llvm-profdata." >&2
    echo "Please ensure LLVM (including llvm-profdata) is installed and accessible." >&2
    echo "You might need to add its 'bin' directory to your PATH environment variable." >&2
    exit 1
fi
if [ -z "$LLVM_COV_CMD" ]; then
    echo "Error: Could not automatically locate llvm-cov." >&2
    echo "Please ensure LLVM (including llvm-cov) is installed and accessible." >&2
    echo "You might need to add its 'bin' directory to your PATH environment variable." >&2
    exit 1
fi

# --- Pre-checks ---
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory '${BUILD_DIR}' not found. Build with Coverage first." >&2
    exit 1
fi
if ! ls ${PROFRAW_PATTERN} 1> /dev/null 2>&1; then
     echo "Error: No '.profraw' files found in '${BUILD_DIR}/tests/cpp/'. Run ctest first." >&2
     exit 1
fi
if [ ! -f "${TEST_EXE_PATH}" ]; then
    echo "Error: Test executable not found at '${TEST_EXE_PATH}'. Check CMake build output location." >&2
    exit 1
fi

# --- Generate and Display Coverage Report ---
echo "Generating terminal coverage report..."

# Merge profraw files (silence stdout on success)
${LLVM_PROFDATA_CMD} merge -sparse ${PROFRAW_PATTERN} -o ${PROFDATA_FILE} || exit 1

# Get project root for path shortening
PROJECT_ROOT=$(pwd)
SED_PROJECT_ROOT=$(printf '%s\n' "$PROJECT_ROOT" | sed 's:[][\\/.^$*]:\\&:g')

# Generate report, shorten paths, and format with column -t
(
    ${LLVM_COV_CMD} report \
        --ignore-filename-regex='.*external.*' \
        --ignore-filename-regex='.*tests/cpp.*' \
        --ignore-filename-regex='.*\.hpp' \
        --ignore-filename-regex='/opt/homebrew/.*' \
        --ignore-filename-regex='/Applications/Xcode.app/.*' \
        --show-region-summary=false \
        --show-instantiation-summary=false \
        "${TEST_EXE_PATH}" \
        -instr-profile=${PROFDATA_FILE} | \
    sed "s|^${SED_PROJECT_ROOT}/||"
) | column -t # Pipe the result of llvm-cov | sed into column -t

echo
echo "--- Coverage generation complete ---"
