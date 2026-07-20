#!/usr/bin/env bash
# Local regression tests for the read-only RDMA readiness preflight.

set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/../check-rdma-readiness.sh"
MOCK_COMMAND="$SCRIPT_DIR/mock-rdma-command.sh"
PYTHON_BIN="$(command -v python3 || true)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/qw3-rdma-readiness-tests.XXXXXX")"
REAL_PATH="/usr/bin:/bin:/usr/sbin:/sbin"
CASE_NUMBER=0
FAILURES=0

trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf '[FAIL] %s\n' "$*" >&2
  FAILURES=$((FAILURES + 1))
}

pass() {
  printf '[PASS] %s\n' "$*"
}

require_equal() {
  local actual="$1" expected="$2" label="$3"
  if [ "$actual" = "$expected" ]; then
    pass "$label"
  else
    fail "$label (expected $expected, got $actual)"
  fi
}

assert_log_omits() {
  local log_file="$1" command_name="$2" label="$3"
  if grep -qx "$command_name" "$log_file" 2>/dev/null; then
    fail "$label (unexpectedly invoked $command_name)"
  else
    pass "$label"
  fi
}

assert_log_count() {
  local log_file="$1" command_name="$2" expected="$3" label="$4" actual
  actual="$(grep -cx "$command_name" "$log_file" 2>/dev/null || true)"
  require_equal "${actual:-0}" "$expected" "$label"
}

assert_peer_rejected() {
  local peer="$1" label="$2"

  run_case ready complete --peer "$peer"
  if [ "$LAST_STATUS" -eq 64 ] && [ ! -s "$LAST_OUTPUT" ] && \
     ! grep -qx route "$LAST_LOG" 2>/dev/null && \
     ! grep -F -- "$peer" "$LAST_ERROR" >/dev/null 2>&1; then
    pass "$label"
  else
    fail "$label (status=$LAST_STATUS, output_bytes=$(wc -c <"$LAST_OUTPUT"))"
  fi
}

run_case() {
  local scenario="$1" sdk_mode="$2"
  shift 2

  CASE_NUMBER=$((CASE_NUMBER + 1))
  CASE_DIR="$TEST_ROOT/case-$CASE_NUMBER"
  MOCK_BIN="$CASE_DIR/bin"
  MOCK_SDK="$CASE_DIR/MacOSX.sdk"
  LAST_OUTPUT="$CASE_DIR/output.json"
  LAST_ERROR="$CASE_DIR/stderr.txt"
  LAST_LOG="$CASE_DIR/calls.log"

  mkdir -p "$MOCK_BIN" "$MOCK_SDK/usr/lib" "$MOCK_SDK/usr/include/infiniband"
  : >"$LAST_LOG"
  if [ "$sdk_mode" = "complete" ]; then
    : >"$MOCK_SDK/usr/lib/librdma.tbd"
    : >"$MOCK_SDK/usr/include/infiniband/verbs.h"
  fi

  for command_name in \
    uname sw_vers system_profiler xcrun ibv_devices ibv_devinfo \
    networksetup ifconfig netstat route; do
    ln -s "$MOCK_COMMAND" "$MOCK_BIN/$command_name"
  done

  PATH="$MOCK_BIN:$REAL_PATH" \
  MOCK_RDMA_SCENARIO="$scenario" \
  MOCK_RDMA_SDK_ROOT="$MOCK_SDK" \
  MOCK_RDMA_CALL_LOG="$LAST_LOG" \
    bash "$TARGET" "$@" >"$LAST_OUTPUT" 2>"$LAST_ERROR"
  LAST_STATUS=$?
}

if [ -z "$PYTHON_BIN" ]; then
  printf 'python3 is required for JSON assertions\n' >&2
  exit 1
fi

bash -n "$TARGET" || exit 1
bash -n "$MOCK_COMMAND" || exit 1

if LC_ALL=C grep -En \
  '(^|[;&|[:space:]])(sudo|rdma_ctl|csrutil|nvram)([;&|[:space:]]|$)|networksetup[[:space:]]+-(set|create|delete)|route[[:space:]]+(add|change|delete|flush)|sysctl[[:space:]]+-?w' \
  "$TARGET" >/dev/null; then
  fail "preflight contains a forbidden state-changing command"
else
  pass "preflight contains no known state-changing command"
fi

run_case ready complete --peer 10.5.0.2
require_equal "$LAST_STATUS" 0 "ready fixture exits 0"
if "$PYTHON_BIN" - "$LAST_OUTPUT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
assert data["overall_status"] == "ready"
assert data["evidence"] == {
    "label": "rdma_preflight_only",
    "hardware_interpretable": False,
    "inference_performance_claim": False,
}
assert data["checks"]["platform"]["apple_silicon"] is True
assert data["checks"]["macos"]["version"] == "26.2.1"
assert data["checks"]["rdma_devices"]["device_count"] == 2
assert data["checks"]["rdma_devices"]["active_port_count"] == 2
assert data["checks"]["thunderbolt_network"]["interface_count"] == 1
assert data["checks"]["thunderbolt_network"]["route_count"] == 2
assert data["checks"]["thunderbolt_network"]["interfaces"] is None
assert data["peer_routes"] == [{
    "target": "redacted",
    "status": "ready",
    "reason": "route_uses_thunderbolt_interface",
    "uses_thunderbolt_interface": True,
    "interface": None,
}]

report = path.read_text()
for secret in (
    "SERIAL-SECRET", "UUID-SECRET", "alice", "10.5.0.1",
    "10.5.0.2", "aa:bb:cc:dd:ee:ff", "fe80::abcd",
):
    assert secret not in report, secret
PY
then
  pass "ready JSON is valid, complete, and redacted"
else
  fail "ready JSON is valid, complete, and redacted"
fi

run_case ready complete \
  --peer 192.0.2.10 \
  --peer 2001:db8::10 \
  --peer fe80::10%12 \
  --peer 2001:db8:0:1:2:3:4:5 \
  --peer ::ffff:192.0.2.128
require_equal "$LAST_STATUS" 0 "numeric IPv4, IPv6, and scoped IPv6 exit 0"
if "$PYTHON_BIN" - "$LAST_OUTPUT" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert len(data["peer_routes"]) == 5
assert all(route["target"] == "redacted" for route in data["peer_routes"])
assert all(route["uses_thunderbolt_interface"] is True for route in data["peer_routes"])
PY
then
  pass "valid numeric peer forms preserve JSON and redaction"
else
  fail "valid numeric peer forms preserve JSON and redaction"
fi
assert_log_count "$LAST_LOG" route 5 "each valid numeric peer reaches route lookup once"

assert_peer_rejected max-1.local "mDNS-style hostname is rejected before route lookup"
assert_peer_rejected localhost "bare hostname is rejected before route lookup"
assert_peer_rejected 10.5.0.2.example "hostname-like dotted value is rejected before route lookup"
assert_peer_rejected fe80::1%en0 "named IPv6 scope is rejected before route lookup"
assert_peer_rejected 999.1.1.1 "out-of-range IPv4 is rejected before route lookup"
assert_peer_rejected 010.5.0.2 "ambiguous leading-zero IPv4 is rejected before route lookup"
assert_peer_rejected 2001:::1 "malformed IPv6 is rejected before route lookup"

run_case ready complete --peer 10.5.0.2 --include-network-identifiers
require_equal "$LAST_STATUS" 0 "explicit identifier fixture exits 0"
if "$PYTHON_BIN" - "$LAST_OUTPUT" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["redaction"]["network_identifiers_redacted"] is False
assert data["checks"]["thunderbolt_network"]["interfaces"] == ["bridge0"]
assert data["peer_routes"][0]["target"] == "10.5.0.2"
assert data["peer_routes"][0]["interface"] == "bridge0"
PY
then
  pass "network identifiers require explicit opt-in"
else
  fail "network identifiers require explicit opt-in"
fi

run_case wrong_route complete --peer 10.5.0.2
require_equal "$LAST_STATUS" 0 "non-Thunderbolt peer route does not change RDMA readiness"
if "$PYTHON_BIN" - "$LAST_OUTPUT" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["overall_status"] == "ready"
assert data["peer_routes"][0]["status"] == "not_ready"
assert data["peer_routes"][0]["uses_thunderbolt_interface"] is False
PY
then
  pass "peer route evidence remains informational"
else
  fail "peer route evidence remains informational"
fi

run_case old_os complete --peer 10.5.0.2
require_equal "$LAST_STATUS" 1 "older macOS fixture exits 1"
if "$PYTHON_BIN" - "$LAST_OUTPUT" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["overall_status"] == "not_ready"
assert data["checks"]["macos"]["reason"] == "version_below_26_2"
assert data["checks"]["rdma_devices"]["reason"] == "unsupported_platform_or_os"
PY
then
  pass "older macOS fails closed with structured evidence"
else
  fail "older macOS fails closed with structured evidence"
fi
assert_log_omits "$LAST_LOG" system_profiler "older macOS skips hardware profiler"
assert_log_omits "$LAST_LOG" ibv_devices "older macOS skips ibv inventory"
assert_log_omits "$LAST_LOG" route "older macOS skips route lookup"

run_case non_apple complete
require_equal "$LAST_STATUS" 1 "non-Apple fixture exits 1"
if "$PYTHON_BIN" - "$LAST_OUTPUT" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["overall_status"] == "not_ready"
assert data["checks"]["platform"]["status"] == "not_ready"
assert data["checks"]["platform"]["apple_silicon"] is False
PY
then
  pass "non-Apple host fails closed with structured evidence"
else
  fail "non-Apple host fails closed with structured evidence"
fi
assert_log_omits "$LAST_LOG" sw_vers "non-Apple host skips macOS inspection"
assert_log_omits "$LAST_LOG" ibv_devices "non-Apple host skips ibv inventory"
assert_log_omits "$LAST_LOG" networksetup "non-Apple host skips network inventory"

run_case unknown complete --peer 10.5.0.2
require_equal "$LAST_STATUS" 2 "uninspectable fixture exits 2"
if "$PYTHON_BIN" - "$LAST_OUTPUT" "$LAST_ERROR" <<'PY'
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
stderr = pathlib.Path(sys.argv[2])
data = json.loads(output.read_text())
assert data["overall_status"] == "unknown"
assert data["checks"]["thunderbolt5"]["status"] == "unknown"
assert data["checks"]["rdma_devices"]["status"] == "unknown"
combined = output.read_text() + stderr.read_text()
for secret in ("SERIAL-SECRET", "UUID-SECRET", "alice", "10.5.0.1"):
    assert secret not in combined, secret
PY
then
  pass "failed probes remain unknown and do not leak stderr"
else
  fail "failed probes remain unknown and do not leak stderr"
fi

run_case ports_down complete
require_equal "$LAST_STATUS" 1 "inactive RDMA ports fixture exits 1"
if "$PYTHON_BIN" - "$LAST_OUTPUT" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["overall_status"] == "not_ready"
assert data["checks"]["rdma_devices"]["status"] == "not_ready"
assert data["checks"]["rdma_devices"]["reason"] == "no_active_rdma_ports"
assert data["checks"]["rdma_devices"]["active_port_count"] == 0
PY
then
  pass "inactive RDMA ports are not_ready"
else
  fail "inactive RDMA ports are not_ready"
fi

run_case ready missing
require_equal "$LAST_STATUS" 1 "missing SDK artifacts fixture exits 1"
if "$PYTHON_BIN" - "$LAST_OUTPUT" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["checks"]["rdma_api"]["status"] == "not_ready"
assert data["checks"]["rdma_api"]["reason"] == "link_library_or_header_missing"
PY
then
  pass "missing SDK RDMA artifacts are not_ready"
else
  fail "missing SDK RDMA artifacts are not_ready"
fi

if [ "$FAILURES" -ne 0 ]; then
  printf '%s test assertion(s) failed\n' "$FAILURES" >&2
  exit 1
fi

printf 'All RDMA readiness tests passed.\n'
