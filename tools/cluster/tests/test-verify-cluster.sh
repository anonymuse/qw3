#!/usr/bin/env bash
set -euo pipefail

TEST_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERIFY="$TEST_DIR/../verify-cluster.sh"
FIXTURE_BIN="$TEST_DIR/fixtures/bin"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/qw3-verify-test.XXXXXX")"
SOURCE_REPO="$TEST_ROOT/source"
ORIGIN_REPO="$TEST_ROOT/origin.git"
DRIVER_REPO="$TEST_ROOT/driver"
REMOTE_ROOT="$TEST_ROOT/remotes"
SSH_LOG="$TEST_ROOT/ssh-calls.log"

cleanup() {
  rm -rf -- "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'not ok - %s\n' "$*" >&2
  exit 1
}

assert_contains() {
  local haystack="$1" needle="$2"
  [[ "$haystack" == *"$needle"* ]] || fail "expected output to contain: $needle"
}

run_verifier() {
  local fail_step="${1:-}"
  PATH="$FIXTURE_BIN:$PATH" \
  TEST_REMOTE_ROOT="$REMOTE_ROOT" \
  MOCK_SSH_CALL_LOG="$SSH_LOG" \
  MOCK_ZIG_FAIL_STEP="$fail_step" \
  DS5_VERIFY_REPO_ROOT="$DRIVER_REPO" \
  DS5_VERIFY_REPO_URL="$ORIGIN_REPO" \
  DS5_VERIFY_SSH_BIN="$FIXTURE_BIN/ssh" \
    "$VERIFY"
}

git init -q --initial-branch=main "$SOURCE_REPO"
printf 'first\n' > "$SOURCE_REPO/payload.txt"
git -C "$SOURCE_REPO" add payload.txt
git -C "$SOURCE_REPO" \
  -c user.name='Verifier Test' -c user.email='verifier@example.invalid' \
  commit -qm 'first fixture commit'
OLD_SHA="$(git -C "$SOURCE_REPO" rev-parse HEAD)"

printf 'second\n' > "$SOURCE_REPO/payload.txt"
git -C "$SOURCE_REPO" add payload.txt
git -C "$SOURCE_REPO" \
  -c user.name='Verifier Test' -c user.email='verifier@example.invalid' \
  commit -qm 'intended fixture commit'
INTENDED_SHA="$(git -C "$SOURCE_REPO" rev-parse HEAD)"

git clone -q --bare "$SOURCE_REPO" "$ORIGIN_REPO"
git clone -q "$ORIGIN_REPO" "$DRIVER_REPO"

for host in pro-1.local max-1.local max-2.local; do
  remote_repo="$REMOTE_ROOT/$host/Code/qw3"
  mkdir -p "$(dirname "$remote_repo")"
  git clone -q "$ORIGIN_REPO" "$remote_repo"
  git -C "$remote_repo" checkout -q --detach "$OLD_SHA"
done

# Happy path: each old checkout is fetched and detached at the driver's exact
# SHA before that node's tests, all three nodes' exit-zero tests pass, and the
# aggregate result is explicitly LAN-only.
set +e
happy_output="$(run_verifier 2>&1)"
happy_status=$?
set -e
[ "$happy_status" -eq 0 ] || { printf '%s\n' "$happy_output" >&2; fail "happy path returned $happy_status"; }
assert_contains "$happy_output" "Intended cluster Git SHA: $INTENDED_SHA"
assert_contains "$happy_output" "exact SHA + clean CPU/GPU tests passed on all 3 nodes"
assert_contains "$happy_output" "No Thunderbolt evidence was collected."
for host in pro-1.local max-1.local max-2.local; do
  actual_sha="$(git -C "$REMOTE_ROOT/$host/Code/qw3" rev-parse HEAD)"
  [ "$actual_sha" = "$INTENDED_SHA" ] || fail "$host remained at $actual_sha"
  [ -z "$(git -C "$REMOTE_ROOT/$host/Code/qw3" status --porcelain=v1 --untracked-files=normal)" ] \
    || fail "$host was not clean after the happy path"
done
printf 'ok - exact SHA and clean-state happy path\n'

# Regression: passing-looking summary text with a nonzero command status must
# fail, retain the raw marker, and make the aggregate result nonzero.
set +e
failure_output="$(run_verifier test-metal 2>&1)"
failure_status=$?
set -e
[ "$failure_status" -ne 0 ] || fail "nonzero Zig exit was accepted as a pass"
assert_contains "$failure_output" "Build Summary: 999/999 tests passed"
assert_contains "$failure_output" "RAW_ZIG_FAILURE_MARKER step=test-metal"
assert_contains "$failure_output" "test-metal FAILED (exit 23)"
assert_contains "$failure_output" "check(s) failed"
printf 'ok - exit code overrides passing-looking summary text\n'

# Regression: a dirty node must be refused before fetch/checkout/tests, and its
# user file and prior HEAD must remain untouched.
DIRTY_REPO="$REMOTE_ROOT/pro-1.local/Code/qw3"
printf 'user work\n' > "$DIRTY_REPO/LOCAL-USER-WORK.txt"
dirty_head="$(git -C "$DIRTY_REPO" rev-parse HEAD)"
set +e
dirty_output="$(run_verifier 2>&1)"
dirty_status=$?
set -e
[ "$dirty_status" -ne 0 ] || fail "dirty remote checkout was accepted"
assert_contains "$dirty_output" "A: checkout is not clean; refusing to update or test"
assert_contains "$dirty_output" "?? LOCAL-USER-WORK.txt"
[ -f "$DIRTY_REPO/LOCAL-USER-WORK.txt" ] || fail "dirty fixture was overwritten"
[ "$(git -C "$DIRTY_REPO" rev-parse HEAD)" = "$dirty_head" ] || fail "dirty checkout HEAD changed"
printf 'ok - dirty remote checkout fails without overwrite\n'

# Regression: fetch failure must be visible and must prevent that node's tests.
rm -- "$DIRTY_REPO/LOCAL-USER-WORK.txt"
BROKEN_REPO="$REMOTE_ROOT/max-1.local/Code/qw3"
git -C "$BROKEN_REPO" remote set-url origin "$TEST_ROOT/missing-origin.git"
set +e
fetch_output="$(run_verifier 2>&1)"
fetch_status=$?
set -e
[ "$fetch_status" -ne 0 ] || fail "failed fetch was masked"
assert_contains "$fetch_output" "fatal:"
assert_contains "$fetch_output" "B: git fetch failed; checkout was not tested"
printf 'ok - fetch failure is raw and fail closed\n'

# The driver itself must be exact and clean before any SSH call is attempted.
: > "$SSH_LOG"
printf 'driver work\n' > "$DRIVER_REPO/LOCAL-DRIVER-WORK.txt"
set +e
driver_output="$(run_verifier 2>&1)"
driver_status=$?
set -e
[ "$driver_status" -ne 0 ] || fail "dirty driver checkout was accepted"
assert_contains "$driver_output" "Driver preflight failed; no cluster node was contacted."
[ ! -s "$SSH_LOG" ] || fail "SSH was called after driver preflight failure"
printf 'ok - dirty driver stops before cluster contact\n'

if grep -nE '\|\|[[:space:]]+true' "$VERIFY"; then
  fail "verifier still masks a failure with || true"
fi
if grep -nE 'grep.*Build Summary' "$VERIFY"; then
  fail "verifier still classifies tests by Build Summary text"
fi
[ -x "$VERIFY" ] || fail "verifier lost its executable bit"
printf 'ok - static guards reject masked failures, summary grepping, and mode loss\n'
