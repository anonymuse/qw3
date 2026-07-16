#!/usr/bin/env bash
# DS5 cluster — verify all 3 nodes have the qw3 repo, pass `zig build test` /
# `test-metal` / `test-gpu`, and can reach each other on the LAN.
#
# Run from the PRIMARY node (Node A / pro-1, Pattern A) — or from Node D, the
# dev/management laptop enrolled via tools/cluster/enroll-dev-node.sh
# (Pattern B; see topology.md) — after tools/cluster/setup-ssh-mesh.sh has
# established the passwordless SSH mesh. Every cluster node (A, B, and C) is
# always tested over SSH like the others, unless this script happens to be
# running on that very node — none of the 3 nodes have passwordless SSH to
# themselves (see setup-ssh-mesh.sh), so a node's result is always that
# node's own checkout, never whichever machine invoked the script. "Am I
# this node?" is gated in exactly one place (on_node, below) rather than
# duplicated per node — see docs/orchestration/LESSONS.md for why that's the
# point. Idempotent / re-runnable: clones if missing, otherwise
# fast-forward pulls.
#
# test-metal and test-gpu are GPU-dependent (need a real Apple Silicon GPU,
# which all 3 nodes have) and are exactly the steps this cluster exists to
# validate — a CPU-only `zig build test` pass does not confirm the
# distributed-inference GPU path actually works on the hardware.

set -uo pipefail

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }

REPO_URL="https://github.com/anonymuse/qw3.git"
SSH="ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"
ZIG_STEPS="test test-metal test-gpu"

A_USER="jesse"; A_HOST="pro-1.local"
B_USER="jesse"; B_HOST="max-1.local"
C_USER="jesse"; C_HOST="max-2.local"

# Short (mDNS-prefix, no ".local") hostnames, for comparing against this
# machine's own LocalHostName in on_node() below.
A_SHORT="${A_HOST%.local}"
B_SHORT="${B_HOST%.local}"
C_SHORT="${C_HOST%.local}"

# Canonical remote clone location per tools/cluster/bootstrap.sh (the ~/Code
# reorg). A node's own local checkout (build_and_test_local) is wherever
# this script lives instead, since that's often a developer-chosen path
# rather than a bootstrap-managed one.
REMOTE_REPO_DIR="\$HOME/Code/qw3"

FAILS=0

# on_node <short-hostname> — true if this machine IS the cluster node named
# by <short-hostname> (one of $A_SHORT/$B_SHORT/$C_SHORT), false otherwise
# (e.g. Node D, the dev laptop enrolled via enroll-dev-node.sh, or a
# different cluster node). None of A/B/C are ever given passwordless SSH to
# themselves, so this gates each node's local fast path instead of
# unconditionally SSHing to a node that may in fact be this very machine,
# which would otherwise fail with a spurious permission error unrelated to
# the actual build/test or ping result — Node C (max-2) doubles as a common
# direct-SSH target for other cluster tasks (see CLAUDE.md), so this script
# runs directly on it often enough that the gap is not hypothetical.
#
# Keep this as ONE function covering all three nodes. It was briefly three
# separate on_node_a()/on_node_c() functions, added one at a time as each
# node's copy of the same bug got independently rediscovered and Node B's
# fix was lost in the process — see docs/orchestration/LESSONS.md. The whole
# point of on_node() is that "am I this node?" only needs solving once.
on_node() {
  local target="$1" name
  name="$(scutil --get LocalHostName 2>/dev/null || hostname -s 2>/dev/null)"
  [ "$name" = "$target" ]
}

if on_node "$A_SHORT"; then
  say "Running from Node A (pro-1) — Pattern A coordination"
elif on_node "$B_SHORT"; then
  say "Running from Node B (max-1)"
elif on_node "$C_SHORT"; then
  say "Running from Node C (max-2)"
else
  say "Running from $(scutil --get LocalHostName 2>/dev/null || hostname -s 2>/dev/null) — driving the cluster remotely (Node D / Pattern B if this is the enrolled dev laptop)"
fi

# build_and_test_local <label> — run in the current shell (used when this
# machine IS the node being tested — A, B, or C).
build_and_test_local() {
  local label="$1"
  cd "$(dirname "$0")/../.." || return 1
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git pull --ff-only >/dev/null 2>&1 || warn "local repo: ff-only pull skipped (local changes present)"
  fi
  local step out any_fail=0
  for step in $ZIG_STEPS; do
    out="$(zig build "$step" --summary all 2>&1)"
    echo "--- $label: $step ---"
    echo "$out" | tail -5
    # Zig 0.16 writes the Build Summary to stderr and prints a cosmetic red
    # "failed command:" line for tests that write to stderr on a passing path
    # (see docs/orchestration/HANDOFF.md landmine #1) — the Build Summary line
    # is the truth (both streams merged above), so grep that, not the exit code.
    if echo "$out" | grep -q "Build Summary:.*tests passed"; then
      ok "$label: $step passed"
    else
      warn "$label: $step FAILED"
      any_fail=1
    fi
  done
  return $any_fail
}

# build_and_test_remote <user@host> <label> — clone/pull + build+test over SSH
build_and_test_remote() {
  local dest="$1" label="$2"
  $SSH "$dest" "
    if [ -d $REMOTE_REPO_DIR/.git ]; then
      cd $REMOTE_REPO_DIR && git pull --ff-only >/dev/null 2>&1 || true
    else
      mkdir -p \\\$HOME/Code
      git clone '$REPO_URL' $REMOTE_REPO_DIR >/dev/null 2>&1
    fi
    cd $REMOTE_REPO_DIR
    any_fail=0
    for step in $ZIG_STEPS; do
      out=\"\$(zig build \$step --summary all 2>&1)\"
      echo \"--- $label: \$step ---\"
      echo \"\$out\" | tail -5
      if echo \"\$out\" | grep -q 'Build Summary:.*tests passed'; then
        echo '[ok] $label: '\$step' passed'
      else
        echo '[warn] $label: '\$step' FAILED'
        any_fail=1
      fi
    done
    exit \$any_fail
  "
}

# build_and_test_node <short-hostname> <user@host> <label> — build+test
# whichever node <short-hostname> names: locally (via build_and_test_local)
# when this machine IS that node, otherwise over SSH (via
# build_and_test_remote) exactly like any other node would be tested. One
# dispatcher shared by A/B/C so the "am I this node" gate only lives in
# on_node(), never duplicated per call site.
build_and_test_node() {
  local short="$1" dest="$2" label="$3"
  if on_node "$short"; then
    say "Node $label ($short, local): clone/pull + zig build test / test-metal / test-gpu"
    build_and_test_local "$label"
  else
    say "Node $label (${dest#*@}): clone/pull + zig build test / test-metal / test-gpu"
    build_and_test_remote "$dest" "$label"
  fi
}

if build_and_test_node "$A_SHORT" "$A_USER@$A_HOST" "A"; then ok "A: all steps passed"; else warn "A: one or more steps FAILED"; FAILS=$((FAILS+1)); fi
if build_and_test_node "$B_SHORT" "$B_USER@$B_HOST" "B"; then ok "B: all steps passed"; else warn "B: one or more steps FAILED"; FAILS=$((FAILS+1)); fi
if build_and_test_node "$C_SHORT" "$C_USER@$C_HOST" "C"; then ok "C: all steps passed"; else warn "C: one or more steps FAILED"; FAILS=$((FAILS+1)); fi

say "LAN reachability (ping, 2 packets each)"
# .local mDNS names resolve to IPv6 link-local addresses ahead of IPv4 on this
# LAN, which ping can't route (same root cause as the earlier SSH "No route
# to host"). Resolve to an IPv4 address explicitly first, on whichever host
# is doing the pinging, then ping that.
check_ping() {
  local from_dest="$1" target_host="$2" label="$3"
  local resolve_and_ping='
    ip=$(dscacheutil -q host -a name '"$target_host"' | awk "/^ip_address: /{print \$2; exit}")
    [ -n "$ip" ] && ping -c 2 -t 3 "$ip" >/dev/null 2>&1
  '
  if [ -z "$from_dest" ]; then
    eval "$resolve_and_ping"
  else
    $SSH "$from_dest" "$resolve_and_ping"
  fi
  if [ $? -eq 0 ]; then ok "$label reachable"; else warn "$label FAILED"; FAILS=$((FAILS+1)); fi
}
# ping_from <short-hostname> <user@host> — the check_ping from_dest for a
# ping sourced FROM that node: empty (check_ping evals locally, same as its
# other local uses) when this machine IS that node, otherwise <user@host> so
# check_ping SSHes there first. Same self-SSH gap as build_and_test_node (no
# cluster node can SSH to itself), applied uniformly to all 6 directions so
# a ping "from" a node is never accidentally run on whichever machine
# invoked the script instead.
ping_from() {
  local short="$1" dest="$2"
  if on_node "$short"; then
    printf ''
  else
    printf '%s' "$dest"
  fi
}
check_ping "$(ping_from "$A_SHORT" "$A_USER@$A_HOST")" "$B_HOST" "A -> B"
check_ping "$(ping_from "$A_SHORT" "$A_USER@$A_HOST")" "$C_HOST" "A -> C"
check_ping "$(ping_from "$B_SHORT" "$B_USER@$B_HOST")" "$C_HOST" "B -> C"
check_ping "$(ping_from "$B_SHORT" "$B_USER@$B_HOST")" "$A_HOST" "B -> A"
check_ping "$(ping_from "$C_SHORT" "$C_USER@$C_HOST")" "$B_HOST" "C -> B"
check_ping "$(ping_from "$C_SHORT" "$C_USER@$C_HOST")" "$A_HOST" "C -> A"

if [ "$FAILS" -eq 0 ]; then
  say "Cluster verification complete: all 3 nodes build+test clean (CPU+GPU), full LAN mesh reachable."
else
  warn "$FAILS check(s) failed. Re-run after investigating; this script is idempotent."
  exit 1
fi
