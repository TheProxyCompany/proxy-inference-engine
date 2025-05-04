#!/usr/bin/env bash

# --- Strict Mode ---
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.
set -o pipefail # Exit status of the last command that threw a non-zero exit code is returned.

# --- Configuration ---
BUILD_DIR="build"
# Define path to the single test executable (update if name changes)
TEST_EXECUTABLE_NAME="pie_core_tests"
TEST_EXE_PATH="${BUILD_DIR}/tests/cpp/${TEST_EXECUTABLE_NAME}"
# Define output paths
PROFRAW_DIR="${BUILD_DIR}/tests/cpp" # Directory where .profraw files are expected
PROFDATA_FILE="${BUILD_DIR}/coverage.profdata"
HTML_REPORT_DIR="${BUILD_DIR}/coverage_html_report"

# --- Find LLVM Tools ---
LLVM_PROFDATA_CMD=""
LLVM_COV_CMD=""

# Check standard PATH first
if command -v llvm-profdata &> /dev/null; then LLVM_PROFDATA_CMD=$(command -v llvm-profdata); fi
if command -v llvm-cov &> /dev/null; then LLVM_COV_CMD=$(command -v llvm-cov); fi

# If not found in PATH, check common Homebrew LLVM paths
if [ -z "$LLVM_PROFDATA_CMD" ] || [ -z "$LLVM_COV_CMD" ]; then
    BREW_LLVM_PREFIX=""
    if command -v brew &> /dev/null; then BREW_LLVM_PREFIX=$(brew --prefix llvm 2>/dev/null || true); fi
    if [ -n "$BREW_LLVM_PREFIX" ] && [ -d "$BREW_LLVM_PREFIX/bin" ]; then
        if [ -z "$LLVM_PROFDATA_CMD" ] && [ -x "$BREW_LLVM_PREFIX/bin/llvm-profdata" ]; then LLVM_PROFDATA_CMD="$BREW_LLVM_PREFIX/bin/llvm-profdata"; fi
        if [ -z "$LLVM_COV_CMD" ] && [ -x "$BREW_LLVM_PREFIX/bin/llvm-cov" ]; then LLVM_COV_CMD="$BREW_LLVM_PREFIX/bin/llvm-cov"; fi
    fi
fi

# Final check
if [ -z "$LLVM_PROFDATA_CMD" ]; then echo "Error: llvm-profdata not found." >&2; exit 1; fi
if [ -z "$LLVM_COV_CMD" ]; then echo "Error: llvm-cov not found." >&2; exit 1; fi

# --- Ensure Build Directory Exists ---
if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory '${BUILD_DIR}' not found. Build with Coverage first." >&2
    exit 1
fi

# --- Run Tests using CTest ---
echo "Running tests with ctest..."
(cd "${BUILD_DIR}" && ctest -C Coverage -V) || {
    echo "Error: ctest execution failed." >&2
    exit 1
}
echo "ctest finished."

# --- Check for Profraw Files ---
# Use find to handle potential spaces and ensure at least one exists
if ! find "${PROFRAW_DIR}" -maxdepth 1 -name '*.profraw' -print -quit | grep -q .; then
    echo "Error: No '.profraw' files found in '${PROFRAW_DIR}/'. Ensure tests ran and generated coverage data." >&2
    exit 1
fi

# --- Check for Test Executable ---
if [ ! -f "${TEST_EXE_PATH}" ]; then
    echo "Error: Test executable not found at '${TEST_EXE_PATH}'. Check CMake build output location." >&2
    exit 1
fi

# --- Generate Coverage Report ---
echo "Merging coverage data..."
${LLVM_PROFDATA_CMD} merge -sparse "${PROFRAW_DIR}"/*.profraw -o "${PROFDATA_FILE}" || {
    echo "Error: llvm-profdata merge failed." >&2
    exit 1
}

echo "Generating HTML coverage report..."
mkdir -p "${HTML_REPORT_DIR}"
${LLVM_COV_CMD} show \
    "${TEST_EXE_PATH}" \
    -instr-profile="${PROFDATA_FILE}" \
    -format=html \
    -output-dir="${HTML_REPORT_DIR}" \
    --ignore-filename-regex='.*\.hpp' \
    --ignore-filename-regex='.*external.*' \
    --ignore-filename-regex='.*tests/cpp.*' \
    --ignore-filename-regex='/opt/homebrew/.*' \
    --ignore-filename-regex='/Applications/Xcode.app/.*' \
    --show-line-counts-or-regions \
    --show-instantiations || {
    echo "Error: llvm-cov show failed." >&2
    exit 1
}

# --- Generate Text Report ---
echo "Generating text summary report..."
${LLVM_COV_CMD} report \
    "${TEST_EXE_PATH}" \
    -instr-profile="${PROFDATA_FILE}" \
    --ignore-filename-regex='.*\.hpp' \
    --ignore-filename-regex='.*external.*' \
    --ignore-filename-regex='.*tests/cpp.*' \
    --ignore-filename-regex='/opt/homebrew/.*' \
    --ignore-filename-regex='/Applications/Xcode.app/.*' \
    --show-region-summary=false \
    --show-instantiation-summary=false || {
    echo "Error: llvm-cov report failed." >&2
    exit 1
}

# --- Finish ---
echo
echo "--- HTML Coverage report generated in '${HTML_REPORT_DIR}/index.html' ---"
echo "--- Profdata file preserved at '${PROFDATA_FILE}' for further analysis ---"

exit 0
