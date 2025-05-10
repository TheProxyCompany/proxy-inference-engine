#!/usr/bin/env bash
# Run Proxy Inference Engine
#   If the project has not been built yet,
#   delegate to `./scripts/build_run.sh`
# Usage: ./run.sh

set -euo pipefail
trap 'on_err $LINENO' ERR
trap on_exit EXIT

###############################################################################
# Configuration (override via environment variables if desired)
###############################################################################
BUILD_DIR="${BUILD_DIR:-build}"
MODEL_PATH="${MODEL_PATH:-.models/llama3-8b-instruct}"
VENV_ACTIVATION_SCRIPT="${VENV_ACTIVATION_SCRIPT:-.venv/bin/activate}"
BUILD_RUN_SCRIPT="./build_run.sh"

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

ENGINE_EXECUTABLE() { printf "%s/bin/pie_engine" "$BUILD_DIR"; }

###############################################################################
# Runtime helpers
###############################################################################
activate_venv() {
  [[ -f "$VENV_ACTIVATION_SCRIPT" ]] || \
    err "Python virtual environment activation script not found: '$VENV_ACTIVATION_SCRIPT'"
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATION_SCRIPT"
}

resolve_model_path() {
  local input="$1" output exit_status
  log "Model path: '$input'"

  output=$(python - <<'PY' "$input"
import sys
from proxy_inference_engine import get_model_path

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
}

launch_engine() {
  log "Launching engine"
  exec "$(ENGINE_EXECUTABLE)" --model "$RESOLVED_MODEL_PATH"
}

###############################################################################
# Main
###############################################################################
main() {
  # If build directory is missing, hand off to the build-and-run script.
  if [[ ! -d "$BUILD_DIR" ]]; then
    log "Build directory '$BUILD_DIR' not found; delegating to $BUILD_RUN_SCRIPT"
    exec "$BUILD_RUN_SCRIPT" "$@"
  fi

  # Preconditions for running only
  require python
  activate_venv

  # Validate engine binary
  [[ -x "$(ENGINE_EXECUTABLE)" ]] || \
    err "Engine binary not found or not executable at '$(ENGINE_EXECUTABLE)'"

  resolve_model_path "$MODEL_PATH"
  launch_engine
}

main "$@"
