#!/usr/bin/env bash
# DS5 cluster — require all 3 nodes to use the invoking checkout's exact Git
# commit, pass `zig build test` / `test-metal` / `test-gpu`, and reach each
# other over the ordinary LAN.
#
# Run from the PRIMARY node (Node A / pro-1, Pattern A) — or from Node D, the
# dev/management laptop enrolled via tools/cluster/enroll-dev-node.sh
# (Pattern B; see topology.md) — after tools/cluster/setup-ssh-mesh.sh has
# established the passwordless SSH mesh. Every cluster node (A, B, and C) is
# tested over SSH unless this script is running on that node. None of the 3
# cluster nodes have passwordless SSH to themselves, so on_node() provides one
# shared local fast path for A, B, and C; see docs/orchestration/LESSONS.md.
#
# The invoking checkout is the source of truth. It must be clean. Its full HEAD
# SHA is fetched and checked out detached on every remote node. Each node must
# be clean and report that exact SHA before that node's tests run; nodes are
# processed sequentially, not through a separate all-node preflight barrier. A
# dirty checkout is never overwritten, and overall success requires all three
# nodes to pass checkout/setup/tests. Git/setup failures and test failures are
# reported from their exit codes, with raw command output left intact.
#
# IMPORTANT: the reachability checks below use LAN mDNS/IPv4 and ping only.
# They are not Thunderbolt transport, throughput, latency, or topology evidence.

set -uo pipefail

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="${DS5_VERIFY_REPO_ROOT:-$DEFAULT_REPO_ROOT}"
REPO_URL="${DS5_VERIFY_REPO_URL:-https://github.com/anonymuse/qw3.git}"
REMOTE_REPO_REL="${DS5_VERIFY_REMOTE_REPO_REL:-Code/qw3}"
SSH_BIN="${DS5_VERIFY_SSH_BIN:-ssh}"
SSH_OPTS=(-4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5)
ZIG_STEPS="test test-metal test-gpu"

A_USER="jesse"; A_HOST="pro-1.local"
B_USER="jesse"; B_HOST="max-1.local"
C_USER="jesse"; C_HOST="max-2.local"

# Short (mDNS-prefix, no ".local") hostnames, for comparing against this
# machine's own LocalHostName in on_node() below.
A_SHORT="${A_HOST%.local}"
B_SHORT="${B_HOST%.local}"
C_SHORT="${C_HOST%.local}"

FAILS=0
INTENDED_SHA=""

run_ssh() {
  "$SSH_BIN" "${SSH_OPTS[@]}" "$@"
}

# on_node <short-hostname> — true if this machine IS the cluster node named
# by <short-hostname>, false otherwise. Keep this as one function covering all
# three nodes; LESSONS.md documents the regressions caused by node-specific
# copies of this check.
on_node() {
  local target="$1" name
  name="$(scutil --get LocalHostName 2>/dev/null || hostname -s 2>/dev/null)"
  [ "$name" = "$target" ]
}

# require_exact_clean_checkout <repo-dir> <expected-sha> <label>
# Refuse to test a checkout unless Git can inspect it, HEAD is the intended
# commit, and tracked/untracked state is empty. Dirty-state details are printed
# so a failure is actionable and no user work is silently overwritten.
require_exact_clean_checkout() {
  local repo_dir="$1" expected_sha="$2" label="$3" actual_sha status

  if ! actual_sha="$(git -C "$repo_dir" rev-parse --verify 'HEAD^{commit}')"; then
    warn "$label: could not resolve HEAD in $repo_dir"
    return 1
  fi
  if [ "$actual_sha" != "$expected_sha" ]; then
    warn "$label: checkout SHA mismatch (expected $expected_sha, found $actual_sha)"
    return 1
  fi
  if ! status="$(git -C "$repo_dir" status --porcelain=v1 --untracked-files=normal)"; then
    warn "$label: could not inspect checkout cleanliness"
    return 1
  fi
  if [ -n "$status" ]; then
    warn "$label: checkout is not clean; refusing to update or test"
    printf '%s\n' "$status"
    return 1
  fi

  ok "$label: exact clean checkout $actual_sha"
}

run_zig_steps() {
  local label="$1" step exit_code any_fail=0

  for step in $ZIG_STEPS; do
    printf '\n--- %s: %s (raw output) ---\n' "$label" "$step"
    if zig build "$step" --summary all; then
      ok "$label: $step passed (exit 0)"
    else
      exit_code=$?
      warn "$label: $step FAILED (exit $exit_code)"
      any_fail=1
    fi
  done

  return "$any_fail"
}

# build_and_test_local <label> — used when this process is running on the node
# being tested. The current checkout is never pulled or switched under a live
# script; it must already be the exact clean invoking checkout.
build_and_test_local() (
  local label="$1"

  require_exact_clean_checkout "$REPO_ROOT" "$INTENDED_SHA" "$label" || return 1
  cd "$REPO_ROOT" || return 1
  run_zig_steps "$label"
)

# build_and_test_remote <user@host> <label> — verify clean state, fetch, check
# out the exact intended commit detached, verify it again, then run all tests.
# The quoted heredoc is evaluated by the remote Bash; no setup/test output is
# truncated or hidden.
build_and_test_remote() {
  local dest="$1" label="$2"

  run_ssh "$dest" /bin/bash -s -- \
    "$REMOTE_REPO_REL" "$REPO_URL" "$INTENDED_SHA" "$label" <<'REMOTE_SCRIPT'
set -uo pipefail

repo_rel="$1"
repo_url="$2"
intended_sha="$3"
label="$4"
repo_dir="$HOME/$repo_rel"

fail() {
  printf '[warn] %s\n' "$*" >&2
  exit 1
}

if [ -e "$repo_dir" ] && [ ! -d "$repo_dir/.git" ]; then
  fail "$label: $repo_dir exists but is not a Git checkout"
fi

if [ -d "$repo_dir/.git" ]; then
  cd "$repo_dir" || fail "$label: could not enter $repo_dir"
  status="$(git status --porcelain=v1 --untracked-files=normal)" \
    || fail "$label: could not inspect checkout cleanliness before fetch"
  if [ -n "$status" ]; then
    printf '[warn] %s\n' "$label: checkout is not clean; refusing to update or test" >&2
    printf '%s\n' "$status" >&2
    exit 1
  fi
else
  mkdir -p "$(dirname "$repo_dir")" \
    || fail "$label: could not create parent directory for $repo_dir"
  git clone "$repo_url" "$repo_dir" \
    || fail "$label: git clone failed"
  cd "$repo_dir" || fail "$label: could not enter cloned checkout"
fi

git fetch --prune origin \
  || fail "$label: git fetch failed; checkout was not tested"
git cat-file -e "${intended_sha}^{commit}" \
  || fail "$label: intended commit $intended_sha is unavailable after fetch"
git checkout --detach "$intended_sha" \
  || fail "$label: checkout of intended commit $intended_sha failed"

actual_sha="$(git rev-parse --verify 'HEAD^{commit}')" \
  || fail "$label: could not resolve HEAD after checkout"
[ "$actual_sha" = "$intended_sha" ] \
  || fail "$label: checkout SHA mismatch (expected $intended_sha, found $actual_sha)"

status="$(git status --porcelain=v1 --untracked-files=normal)" \
  || fail "$label: could not inspect checkout cleanliness after checkout"
if [ -n "$status" ]; then
  printf '[warn] %s\n' "$label: checkout is not clean after checkout; refusing to test" >&2
  printf '%s\n' "$status" >&2
  exit 1
fi

printf '[ok] %s\n' "$label: exact clean checkout $actual_sha"

any_fail=0
for step in test test-metal test-gpu; do
  printf '\n--- %s: %s (raw output) ---\n' "$label" "$step"
  if zig build "$step" --summary all; then
    printf '[ok] %s\n' "$label: $step passed (exit 0)"
  else
    exit_code=$?
    printf '[warn] %s\n' "$label: $step FAILED (exit $exit_code)" >&2
    any_fail=1
  fi
done

exit "$any_fail"
REMOTE_SCRIPT
}

# build_and_test_node <short-hostname> <user@host> <label> — use the local
# path for whichever node invokes the script, otherwise use SSH. The general
# dispatcher preserves the all-node self-check fix documented in LESSONS.md.
build_and_test_node() {
  local short="$1" dest="$2" label="$3"
  if on_node "$short"; then
    say "Node $label ($short, local): verify exact SHA + zig build test / test-metal / test-gpu"
    build_and_test_local "$label"
  else
    say "Node $label (${dest#*@}): fetch exact SHA + zig build test / test-metal / test-gpu"
    build_and_test_remote "$dest" "$label"
  fi
}

# check_ping performs only LAN/mDNS reachability. It deliberately leaves the
# resolver/ping output visible on failure and never describes the result as a
# Thunderbolt check.
check_ping() {
  local from_dest="$1" target_host="$2" label="$3"
  local resolve_and_ping=''

  resolve_and_ping='ip=$(dscacheutil -q host -a name '"$target_host"' | awk '\''/^ip_address: /{print $2; exit}'\'')
if [ -z "$ip" ]; then
  echo "could not resolve an IPv4 LAN address for '"$target_host"'" >&2
  exit 1
fi
echo "resolved '"$target_host"' to LAN IPv4 $ip"
ping -c 2 -t 3 "$ip"'

  if [ -z "$from_dest" ]; then
    if /bin/bash -c "$resolve_and_ping"; then
      ok "$label reachable via LAN IPv4"
    else
      warn "$label LAN reachability FAILED"
      FAILS=$((FAILS+1))
    fi
  elif run_ssh "$from_dest" "$resolve_and_ping"; then
    ok "$label reachable via LAN IPv4"
  else
    warn "$label LAN reachability FAILED"
    FAILS=$((FAILS+1))
  fi
}

# ping_from <short-hostname> <user@host> — choose local execution for the node
# running this script, otherwise return its SSH destination. This mirrors the
# generalized self-node behavior used for builds across all six directions.
ping_from() {
  local short="$1" dest="$2"
  if on_node "$short"; then
    printf ''
  else
    printf '%s' "$dest"
  fi
}

main() {
  local driver_name

  FAILS=0
  if ! INTENDED_SHA="$(git -C "$REPO_ROOT" rev-parse --verify 'HEAD^{commit}')"; then
    warn "driver: could not resolve an intended Git commit from $REPO_ROOT"
    return 1
  fi
  if ! require_exact_clean_checkout "$REPO_ROOT" "$INTENDED_SHA" "driver"; then
    warn "Driver preflight failed; no cluster node was contacted."
    return 1
  fi

  say "Intended cluster Git SHA: $INTENDED_SHA"
  if on_node "$A_SHORT"; then
    say "Running from Node A (pro-1) — Pattern A coordination"
  elif on_node "$B_SHORT"; then
    say "Running from Node B (max-1)"
  elif on_node "$C_SHORT"; then
    say "Running from Node C (max-2)"
  else
    driver_name="$(scutil --get LocalHostName 2>/dev/null || hostname -s 2>/dev/null)"
    say "Running from $driver_name — driving the cluster remotely (Node D / Pattern B if enrolled)"
  fi

  if build_and_test_node "$A_SHORT" "$A_USER@$A_HOST" "A"; then ok "A: all steps passed"; else warn "A: checkout/setup/test FAILED"; FAILS=$((FAILS+1)); fi
  if build_and_test_node "$B_SHORT" "$B_USER@$B_HOST" "B"; then ok "B: all steps passed"; else warn "B: checkout/setup/test FAILED"; FAILS=$((FAILS+1)); fi
  if build_and_test_node "$C_SHORT" "$C_USER@$C_HOST" "C"; then ok "C: all steps passed"; else warn "C: checkout/setup/test FAILED"; FAILS=$((FAILS+1)); fi

  say "LAN-only reachability (mDNS IPv4 + ping, 2 packets each; not Thunderbolt evidence)"
  check_ping "$(ping_from "$A_SHORT" "$A_USER@$A_HOST")" "$B_HOST" "A -> B"
  check_ping "$(ping_from "$A_SHORT" "$A_USER@$A_HOST")" "$C_HOST" "A -> C"
  check_ping "$(ping_from "$B_SHORT" "$B_USER@$B_HOST")" "$C_HOST" "B -> C"
  check_ping "$(ping_from "$B_SHORT" "$B_USER@$B_HOST")" "$A_HOST" "B -> A"
  check_ping "$(ping_from "$C_SHORT" "$C_USER@$C_HOST")" "$B_HOST" "C -> B"
  check_ping "$(ping_from "$C_SHORT" "$C_USER@$C_HOST")" "$A_HOST" "C -> A"

  if [ "$FAILS" -eq 0 ]; then
    say "Cluster verification complete: exact SHA + clean CPU/GPU tests passed on all 3 nodes; LAN-only reachability passed. No Thunderbolt evidence was collected."
    return 0
  fi

  warn "$FAILS check(s) failed. No Thunderbolt evidence was collected; investigate the raw output before re-running."
  return 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
