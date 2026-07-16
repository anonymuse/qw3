#!/usr/bin/env bash
# DS5 cluster — enroll THIS machine as Node D (the dev/management laptop, not
# a compute node) with passwordless SSH access into the 3-node compute
# cluster (A/B/C), so it can drive cluster scripts (setup-ssh-mesh.sh,
# verify-cluster.sh, ds5 bench link, etc.) the same way Node A does.
#
# Run FROM Node D (the dev laptop). One-time interactive step per node: each
# prompts for its login password (Remote Login must already be ON —
# bootstrap.sh Phase 0 does this on A/B/C). After that it's idempotent and
# safe to re-run.
#
# This is one-directional (Node D -> cluster only) — the cluster nodes are
# never given a key to reach back into Node D. Node D is also NOT added to
# manifests/cluster/lab.zon (the TB5 inference mesh): this is SSH/admin
# access, not DS5 compute participation. See tools/cluster/topology.md for
# the full access model (Pattern A vs. Pattern B coordination).

set -uo pipefail

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }

# Cluster nodes all run user "jesse" regardless of this machine's local user.
# Override with DS5_CLUSTER_USER=<user> if a future cluster differs.
REMOTE_USER="${DS5_CLUSTER_USER:-jesse}"
NODES=(pro-1.local max-1.local max-2.local)
LABELS=("A (pro-1)" "B (max-1)" "C (max-2)")

SSH="ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"

# 1. Local SSH key ------------------------------------------------------------
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
  say "Generating ed25519 SSH key for Node D (this dev machine)…"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  ssh-keygen -t ed25519 -N "" -C "ds5-dev-$(scutil --get LocalHostName 2>/dev/null || hostname)" -f "$HOME/.ssh/id_ed25519" >/dev/null
fi
PUBKEY="$(cat "$HOME/.ssh/id_ed25519.pub")"
ok "Local pubkey: $PUBKEY"

# 2. Seed authorized_keys on each node (interactive password, first run only) -
say "Pushing this laptop's public key to each cluster node."
say "You'll be prompted for the '$REMOTE_USER' login password on each node the first time."
for host in "${NODES[@]}"; do
  dest="$REMOTE_USER@$host"
  say "Enrolling on $dest"
  if command -v ssh-copy-id >/dev/null 2>&1; then
    ssh-copy-id -i "$HOME/.ssh/id_ed25519.pub" -o StrictHostKeyChecking=accept-new "$dest" \
      || warn "ssh-copy-id failed for $dest — is it awake and on the LAN?"
  else
    ssh -4 -o StrictHostKeyChecking=accept-new "$dest" "
      mkdir -p ~/.ssh && chmod 700 ~/.ssh
      touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
      grep -qxF '$PUBKEY' ~/.ssh/authorized_keys || echo '$PUBKEY' >> ~/.ssh/authorized_keys
    " || warn "key push failed for $dest — is it awake and on the LAN?"
  fi
done

# 3. Verify passwordless SSH in all 3 directions ------------------------------
say "Verifying passwordless SSH from Node D (this dev machine) to all 3 nodes"
FAILS=0
for i in "${!NODES[@]}"; do
  host="${NODES[$i]}"; label="${LABELS[$i]}"
  if $SSH "$REMOTE_USER@$host" true 2>/dev/null; then
    ok "dev -> $label"
  else
    warn "dev -> $label FAILED"
    FAILS=$((FAILS+1))
  fi
done

if [ "$FAILS" -eq 0 ]; then
  say "Node D enrolled: passwordless SSH to all 3 cluster nodes confirmed."
  say "You can now run tools/cluster/verify-cluster.sh (or any cluster script) from this machine."
else
  warn "$FAILS node(s) failed verification. Re-run this script once they're reachable; it's idempotent."
  exit 1
fi
