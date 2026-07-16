#!/usr/bin/env bash
# DS5 cluster — verify all 3 nodes have the qw3 repo, pass `zig build test` /
# `test-metal` / `test-gpu`, and can reach each other on the LAN.
#
# Run from the PRIMARY node (Node A / pro-1) — or from any machine enrolled
# via tools/cluster/enroll-dev-node.sh — after tools/cluster/setup-ssh-mesh.sh
# has established the passwordless SSH mesh. Node A is always tested over SSH
# like B/C unless this script is actually running on Node A itself, so the
# result labeled "Node A" is always Node A's checkout, never the invoking
# machine's. Idempotent / re-runnable: clones if missing, otherwise
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

# Canonical remote clone location per tools/cluster/bootstrap.sh (the ~/Code
# reorg). Node A's own checkout (build_and_test_local) is wherever this
# script lives instead, since that's often a developer-chosen path rather
# than a bootstrap-managed one.
REMOTE_REPO_DIR="\$HOME/Code/qw3"

FAILS=0

# on_node_a — true if this machine IS Node A (pro-1), false if we're some
# other machine (e.g. a dev laptop enrolled via enroll-dev-node.sh) driving
# the cluster remotely. Node A is never given passwordless SSH to itself (see
# setup-ssh-mesh.sh / enroll-dev-node.sh), so this gates the local fast path
# instead of unconditionally SSHing to A, which would break when actually
# run on A.
on_node_a() {
  local name
  name="$(scutil --get LocalHostName 2>/dev/null || hostname -s 2>/dev/null)"
  [ "$name" = "${A_HOST%.local}" ]
}

# build_and_test_local — run in the current shell (used only when this machine IS Node A)
build_and_test_local() {
  cd "$(dirname "$0")/../.." || return 1
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git pull --ff-only >/dev/null 2>&1 || warn "local repo: ff-only pull skipped (local changes present)"
  fi
  local step out any_fail=0
  for step in $ZIG_STEPS; do
    out="$(zig build "$step" --summary all 2>&1)"
    echo "--- A: $step ---"
    echo "$out" | tail -5
    # Zig 0.16 writes the Build Summary to stderr and prints a cosmetic red
    # "failed command:" line for tests that write to stderr on a passing path
    # (see docs/orchestration/HANDOFF.md landmine #1) — the Build Summary line
    # is the truth (both streams merged above), so grep that, not the exit code.
    if echo "$out" | grep -q "Build Summary:.*tests passed"; then
      ok "A: $step passed"
    else
      warn "A: $step FAILED"
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

if on_node_a; then
  say "Node A (pro-1, local): clone/pull + zig build test / test-metal / test-gpu"
  if build_and_test_local; then ok "A: all steps passed"; else warn "A: one or more steps FAILED"; FAILS=$((FAILS+1)); fi
else
  say "Node A ($A_HOST): clone/pull + zig build test / test-metal / test-gpu"
  if build_and_test_remote "$A_USER@$A_HOST" "A"; then ok "A: all steps passed"; else warn "A: one or more steps FAILED"; FAILS=$((FAILS+1)); fi
fi

say "Node B ($B_HOST): clone/pull + zig build test / test-metal / test-gpu"
if build_and_test_remote "$B_USER@$B_HOST" "B"; then ok "B: all steps passed"; else warn "B: one or more steps FAILED"; FAILS=$((FAILS+1)); fi

say "Node C ($C_HOST): clone/pull + zig build test / test-metal / test-gpu"
if build_and_test_remote "$C_USER@$C_HOST" "C"; then ok "C: all steps passed"; else warn "C: one or more steps FAILED"; FAILS=$((FAILS+1)); fi

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
# "A -> B"/"A -> C" ping from wherever Node A actually is: locally when this
# machine IS Node A (empty from_dest, same as check_ping's other local uses),
# otherwise hop over SSH to A first — same on_node_a gate as the build/test
# step above, for the same reason (this machine may not be Node A).
if on_node_a; then
  check_ping ""                 "$B_HOST" "A -> B"
  check_ping ""                 "$C_HOST" "A -> C"
else
  check_ping "$A_USER@$A_HOST"  "$B_HOST" "A -> B"
  check_ping "$A_USER@$A_HOST"  "$C_HOST" "A -> C"
fi
check_ping "$B_USER@$B_HOST"  "$C_HOST" "B -> C"
check_ping "$B_USER@$B_HOST"  "$A_HOST" "B -> A"
check_ping "$C_USER@$C_HOST"  "$B_HOST" "C -> B"
check_ping "$C_USER@$C_HOST"  "$A_HOST" "C -> A"

if [ "$FAILS" -eq 0 ]; then
  say "Cluster verification complete: all 3 nodes build+test clean (CPU+GPU), full LAN mesh reachable."
else
  warn "$FAILS check(s) failed. Re-run after investigating; this script is idempotent."
  exit 1
fi
