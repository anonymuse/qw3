#!/bin/bash
# Deterministic synthetic Metal soak. This does not run a real model or emit
# hardware-interpretable performance evidence.
# Usage: ./tools/run-metal-backend-remote.sh [--node LABEL] [--output-dir DIR]
#        [--iterations N] [--dry-run]

set -euo pipefail

NODE=""
OUTPUT_DIR=""
ITERATIONS=64
DRY_RUN=false
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TEST_COMMAND="zig build test-metal -Doptimize=ReleaseFast --summary all"

usage() {
  cat <<'EOF'
usage: ./tools/run-metal-backend-remote.sh [options]

Options:
  --node LABEL        Record a node label in the report; does not perform SSH.
  --output-dir DIR    Artifact directory.
  --iterations N      Deterministic test repetitions (default: 64).
  --dry-run           Validate options and print the plan without running Zig.
  -h, --help          Show this help.

Evidence boundary:
  evidence_class=synthetic_metal_soak
  hardware_interpretable=false
  real_model=false
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "$2" ]]; then
    echo "ERROR: $1 requires a value" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node)
      require_value "$@"
      NODE="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$@"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --iterations)
      require_value "$@"
      ITERATIONS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --iterations must be a positive integer" >&2
  exit 2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="./bench/results/metal-soak-${TIMESTAMP}"
fi

echo "DS5 deterministic synthetic Metal soak"
echo "evidence_class=synthetic_metal_soak"
echo "hardware_interpretable=false"
echo "real_model=false"
echo "timestamp=${TIMESTAMP}"
echo "node_label=${NODE:-current}"
echo "iterations=${ITERATIONS}"
echo "output_dir=${OUTPUT_DIR}"
echo "test_command=${TEST_COMMAND}"

if [[ "$DRY_RUN" == true ]]; then
  echo "dry_run=true"
  echo "No build, Metal test, or SSH command was executed."
  exit 0
fi

# Capture source provenance before creating artifacts, so the soak output does
# not affect the recorded dirty state. Porcelain paths are held only in a shell
# variable long enough to derive a boolean and are never printed or persisted.
GIT_SHA="unknown"
GIT_DIRTY="unknown"
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  if GIT_SHA_CANDIDATE=$(git rev-parse HEAD 2>/dev/null); then
    GIT_SHA="$GIT_SHA_CANDIDATE"
  fi
  if GIT_STATUS_PORCELAIN=$(git status --porcelain --untracked-files=normal 2>/dev/null); then
    if [[ -n "$GIT_STATUS_PORCELAIN" ]]; then
      GIT_DIRTY="true"
    else
      GIT_DIRTY="false"
    fi
  fi
  unset GIT_SHA_CANDIDATE GIT_STATUS_PORCELAIN
fi

# Never merge a run with prior artifacts. The initial check gives a useful
# diagnostic; the plain mkdir (not mkdir -p) also closes the creation race.
if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
  echo "ERROR: output destination already exists; refusing to overwrite: ${OUTPUT_DIR}" >&2
  exit 2
fi
OUTPUT_PARENT=$(dirname "$OUTPUT_DIR")
mkdir -p "$OUTPUT_PARENT"
if ! mkdir "$OUTPUT_DIR"; then
  echo "ERROR: could not claim new output destination: ${OUTPUT_DIR}" >&2
  exit 2
fi

BUILD_LOG="${OUTPUT_DIR}/build.log"
ITERATION_SUMMARY="${OUTPUT_DIR}/iterations.tsv"
REPORT="${OUTPUT_DIR}/metal-soak-report.txt"
RUN_START=$(date +%s)
PASSED=0
FAILED=0
ATTEMPTED=0
FINAL_STATUS="fail"
FAILURE_STAGE="unknown"

printf "iteration\tstatus\tduration_seconds\tlog\n" > "$ITERATION_SUMMARY"

write_report() {
  local run_end
  local run_duration
  run_end=$(date +%s)
  run_duration=$((run_end - RUN_START))

  cat > "$REPORT" <<EOF
DS5 deterministic synthetic Metal soak
======================================
evidence_class=synthetic_metal_soak
hardware_interpretable=false
real_model=false
status=${FINAL_STATUS}
failure_stage=${FAILURE_STAGE}

timestamp=${TIMESTAMP}
node_label=${NODE:-current}
git_sha=${GIT_SHA}
git_dirty=${GIT_DIRTY}
test_command=${TEST_COMMAND}
iterations=${ITERATIONS}
iterations_attempted=${ATTEMPTED}
passed=${PASSED}
failed=${FAILED}
operational_duration_seconds=${run_duration}

Interpretation:
  A pass means the synthetic Metal test command exited successfully in every
  deterministic repetition. This soak is a stability signal only. It is not a
  real-model inference, generation, quantization, throughput, or speedup result.

Artifacts:
  build_log=${BUILD_LOG}
  iteration_summary=${ITERATION_SUMMARY}
  raw_iteration_logs=${OUTPUT_DIR}/iteration-NNN.log
EOF
}

print_report_location() {
  cat "$REPORT"
  echo "Report saved to: ${REPORT}"
}

echo "Running build preflight; raw output: ${BUILD_LOG}"
if zig build -Doptimize=ReleaseFast --summary all > "$BUILD_LOG" 2>&1; then
  echo "Build preflight passed."
else
  echo "ERROR: build preflight failed; see ${BUILD_LOG}" >&2
  FAILURE_STAGE="build_preflight"
  write_report
  print_report_location
  exit 1
fi

for ((i = 1; i <= ITERATIONS; i++)); do
  ITERATION_LOG=$(printf "%s/iteration-%03d.log" "$OUTPUT_DIR" "$i")
  ITERATION_START=$(date +%s)
  printf "[%d/%d] synthetic Metal test ... " "$i" "$ITERATIONS"

  # Keep the command unpiped: its exit status alone decides pass/fail. Every
  # byte of stdout/stderr is retained in the iteration-specific raw log.
  if zig build test-metal -Doptimize=ReleaseFast --summary all > "$ITERATION_LOG" 2>&1; then
    STATUS="pass"
    PASSED=$((PASSED + 1))
    echo "pass"
  else
    STATUS="fail"
    FAILED=$((FAILED + 1))
    echo "fail (see ${ITERATION_LOG})"
  fi

  ITERATION_END=$(date +%s)
  ATTEMPTED=$i
  printf "%d\t%s\t%d\t%s\n" \
    "$i" "$STATUS" "$((ITERATION_END - ITERATION_START))" "$ITERATION_LOG" \
    >> "$ITERATION_SUMMARY"
done

if [[ "$FAILED" -eq 0 ]]; then
  FINAL_STATUS="pass"
  FAILURE_STAGE="none"
else
  FINAL_STATUS="fail"
  FAILURE_STAGE="test_iterations"
fi

write_report
print_report_location

if [[ "$FAILED" -gt 0 ]]; then
  exit 1
fi
