#!/usr/bin/env bash
# Build and run Proxy Inference Engine
# Usage: ./scripts/build_run.sh

set -euo pipefail
trap 'on_err $LINENO' ERR
trap on_exit EXIT

###############################################################################
# Configuration (override via environment variables if desired)
###############################################################################
BUILD_DIR="${BUILD_DIR:-build}"
BUILD_TYPE="${BUILD_TYPE:-RelWithDebInfo}"
MODEL_PATH="${MODEL_PATH:-.models/llama3-8b-instruct}"
VENV_ACTIVATION_SCRIPT="${VENV_ACTIVATION_SCRIPT:-.venv/bin/activate}"

###############################################################################
# Helpers
###############################################################################
log()   { printf "[%s] %s\n" "$(date +'%Y-%m-%d %H:%M:%S')" "$*"; }
err()   { log "ERROR: $*"; exit 1; }

on_err() {
  local line="$1" code="$?"
  err "Command failed with exit code $code at line $line"
}

on_exit() {
  local code="$?"
  [[ $code -eq 0 ]] && log "Completed successfully" \
                    || log "Exited with code $code"
}

require() {
  command -v "$1" &>/dev/null || err "Required command '$1' not found"
}

cores() {
  command -v nproc &>/dev/null && nproc || sysctl -n hw.ncpu
}

###############################################################################
# Build pipeline
###############################################################################
build_project() {
  log "Removing old build directory"
  rm -rf "$BUILD_DIR"

  log "Configuring project ($BUILD_TYPE)"
  cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

  log "Building with $(cores) parallel jobs"
  cmake --build "$BUILD_DIR" --parallel "$(cores)"

  log "Installing into $BUILD_DIR"
  cmake --install "$BUILD_DIR" --prefix "$BUILD_DIR"
}

###############################################################################
# Engine runtime helpers
###############################################################################
ENGINE_EXECUTABLE() { printf "%s/bin/pie_engine" "$BUILD_DIR"; }

validate_engine() {
  local exe
  exe="$(ENGINE_EXECUTABLE)"
  [[ -x "$exe" ]] || err "Engine binary not found or not executable at '$exe'"
  log "Validated engine executable: '$exe'"
}

activate_venv() {
  [[ -f "$VENV_ACTIVATION_SCRIPT" ]] || \
    err "Python virtual environment activation script not found: '$VENV_ACTIVATION_SCRIPT'"
  log "Activating Python virtual environment"
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATION_SCRIPT"
}

resolve_model_path() {
  local input="$1" output exit_status
  log "Resolving model path for input: '$input'"

  # Inline Python to resolve model path
  output=$(python - <<'PY' "$input"
import sys
from proxy_inference_engine.models.utils import get_model_path

if len(sys.argv) != 2:
    sys.stderr.write("Expected exactly one argument (model path)\n")
    sys.exit(1)

print(get_model_path(sys.argv[1]))
PY
  )
  exit_status=$?

  [[ $exit_status -eq 0 ]] || err "Python model resolution utility failed (exit $exit_status)"
  [[ -n "$output" ]]       || err "Python utility returned empty output"
  [[ -d "$output" ]]       || err "Resolved model path '$output' is not a directory"

  RESOLVED_MODEL_PATH="$output"
  log "Model resolved to: '$RESOLVED_MODEL_PATH'"
}

launch_engine() {
  log "Launching engine"
  exec "$(ENGINE_EXECUTABLE)" --model "$RESOLVED_MODEL_PATH"
}

###############################################################################
# Main
###############################################################################
main() {
  require cmake
  require python

  build_project
  validate_engine
  activate_venv
  resolve_model_path "$MODEL_PATH"
  launch_engine
}

main "$@"
