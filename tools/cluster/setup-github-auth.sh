#!/usr/bin/env bash
# DS5 cluster — set up GitHub SSH authentication for all 3 nodes (A/B/C).
#
# Generates a GitHub-specific ed25519 SSH key on each node (separate from the
# cluster-mesh key), gathers public keys, converts git remotes from HTTPS to SSH,
# sets per-node git identity, and verifies connectivity to GitHub. The human
# must then manually add each node's public key to GitHub as a Deploy Key via
# Settings → Deploy keys (the script displays them all for easy copying).
#
# Run FROM Node A (or Node D if enrolled). Idempotent: safe to re-run.
# Pass --verify-only to re-check everything without repeating setup.
#
# This approach separates GitHub auth keys from cluster-mesh keys for better
# security isolation: compromise of one doesn't compromise the other.

set -uo pipefail

VERIFY_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --verify-only) VERIFY_ONLY=1 ;;
  esac
done

say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }

SSH="ssh -4 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5"

# Cluster node addresses (hardcoded; matches setup-ssh-mesh.sh and enroll-dev-node.sh)
A_USER="jesse"; A_HOST="pro-1.local"
B_USER="jesse"; B_HOST="max-1.local"
C_USER="jesse"; C_HOST="max-2.local"
NODES_ARRAY=(
  "$A_USER@$A_HOST:A (pro-1)"
  "$B_USER@$B_HOST:B (max-1)"
  "$C_USER@$C_HOST:C (max-2)"
)

# Repo directory (same on all nodes)
REPO_DIR="$HOME/Code/qw3"

# GitHub SSH config to add to ~/.ssh/config on each node
GITHUB_SSH_CONFIG="
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  AddKeysToAgent yes
  StrictHostKeyChecking accept-new
"

# generate_github_key <user@host> — idempotently generate ~/.ssh/id_ed25519_github on a remote node
generate_github_key() {
  local dest="$1" host="${dest#*@}"
  if $SSH "$dest" "[ -f ~/.ssh/id_ed25519_github ]" 2>/dev/null; then
    return 0  # key already exists
  fi
  $SSH "$dest" "
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    ssh-keygen -t ed25519 -N '' -C 'ds5-github-${host%.local}' -f ~/.ssh/id_ed25519_github >/dev/null
  " || return 1
}

# ensure_ssh_config <user@host> — add GitHub Host entry to ~/.ssh/config if not present
ensure_ssh_config() {
  local dest="$1"
  $SSH "$dest" "
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    if ! grep -q '^Host github.com' ~/.ssh/config 2>/dev/null; then
      cat >> ~/.ssh/config <<'EOFCONFIG'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  AddKeysToAgent yes
  StrictHostKeyChecking accept-new
EOFCONFIG
      chmod 600 ~/.ssh/config
    fi
  " || return 1
}

# convert_remote_to_ssh <user@host> <repo_dir> — convert git remote from HTTPS to SSH
convert_remote_to_ssh() {
  local dest="$1" repo_dir="$2"
  $SSH "$dest" "
    cd '$repo_dir' 2>/dev/null || exit 1
    current_remote=\$(git config --get remote.origin.url 2>/dev/null || echo '')
    if [[ \"\$current_remote\" == https://github.com/* ]]; then
      # Extract owner/repo from HTTPS URL: https://github.com/owner/repo.git → git@github.com:owner/repo.git
      new_remote=\"\${current_remote#https://github.com/}\"
      new_remote=\"git@github.com:\${new_remote}\"
      git remote set-url origin \"\$new_remote\"
    fi
  " || return 1
}

# get_github_pubkey <user@host> — retrieve ~/.ssh/id_ed25519_github.pub from a remote node
get_github_pubkey() {
  local dest="$1"
  $SSH "$dest" "cat ~/.ssh/id_ed25519_github.pub 2>/dev/null" || return 1
}

# set_git_identity <user@host> <repo_dir> — set git user.name and user.email locally in the repo
set_git_identity() {
  local dest="$1" repo_dir="$2"
  $SSH "$dest" "
    cd '$repo_dir' 2>/dev/null || exit 1
    git config user.name 'DS5 Cluster' 2>/dev/null || true
    git config user.email 'ds5@cluster.local' 2>/dev/null || true
  " || return 1
}

# verify_github_ssh <user@host> — test SSH connectivity to github.com
verify_github_ssh() {
  local dest="$1"
  if $SSH "$dest" "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519_github git@github.com" true 2>/dev/null; then
    return 0
  else
    # GitHub returns a non-zero exit code even on successful auth, so we check the output instead
    local output
    output=$($SSH "$dest" "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519_github git@github.com 2>&1" || true)
    if echo "$output" | grep -q "successfully authenticated"; then
      return 0
    fi
    return 1
  fi
}

# ============================================================================

if [ "$VERIFY_ONLY" = 0 ]; then
  say "Generating GitHub SSH keys on all 3 nodes…"
  for node_spec in "${NODES_ARRAY[@]}"; do
    node_dest="${node_spec%%:*}"
    node_label="${node_spec#*:}"
    if generate_github_key "$node_dest"; then
      ok "Node $node_label: GitHub key generated/exists"
    else
      warn "Node $node_label: Failed to generate key"
      exit 1
    fi
  done

  say "Adding SSH config entries for GitHub…"
  for node_spec in "${NODES_ARRAY[@]}"; do
    node_dest="${node_spec%%:*}"
    node_label="${node_spec#*:}"
    if ensure_ssh_config "$node_dest"; then
      ok "Node $node_label: SSH config updated"
    else
      warn "Node $node_label: Failed to update SSH config"
      exit 1
    fi
  done

  say "Converting git remotes to SSH…"
  for node_spec in "${NODES_ARRAY[@]}"; do
    node_dest="${node_spec%%:*}"
    node_label="${node_spec#*:}"
    if convert_remote_to_ssh "$node_dest" "$REPO_DIR"; then
      ok "Node $node_label: Git remote converted"
    else
      warn "Node $node_label: Failed to convert remote"
      exit 1
    fi
  done

  say "Setting git user identity…"
  for node_spec in "${NODES_ARRAY[@]}"; do
    node_dest="${node_spec%%:*}"
    node_label="${node_spec#*:}"
    if set_git_identity "$node_dest" "$REPO_DIR"; then
      ok "Node $node_label: Git identity set"
    else
      warn "Node $node_label: Failed to set git identity"
      exit 1
    fi
  done
fi

# ============================================================================
# Gather and display public keys
# ============================================================================

say "Gathering GitHub public keys…"
declare -A PUBKEYS
FAILURES=0

for node_spec in "${NODES_ARRAY[@]}"; do
  node_dest="${node_spec%%:*}"
  node_label="${node_spec#*:}"
  pubkey=$(get_github_pubkey "$node_dest")
  if [ -n "$pubkey" ]; then
    PUBKEYS["$node_label"]="$pubkey"
    ok "Node $node_label: $pubkey"
  else
    warn "Node $node_label: Failed to retrieve public key"
    FAILURES=$((FAILURES+1))
  fi
done

if [ "$FAILURES" -gt 0 ]; then
  warn "Failed to retrieve $FAILURES key(s). Check node connectivity and re-run."
  exit 1
fi

# ============================================================================
# Verify git remote and SSH connectivity
# ============================================================================

say "Verifying git remote URLs…"
VERIFY_FAILURES=0

for node_spec in "${NODES_ARRAY[@]}"; do
  node_dest="${node_spec%%:*}"
  node_label="${node_spec#*:}"
  remote_url=$($SSH "$node_dest" "cd '$REPO_DIR' && git config --get remote.origin.url 2>/dev/null" || echo "ERROR")
  if [[ "$remote_url" == git@github.com:* ]]; then
    ok "Node $node_label: remote is SSH"
  else
    warn "Node $node_label: remote is not SSH (got: $remote_url)"
    VERIFY_FAILURES=$((VERIFY_FAILURES+1))
  fi
done

say "Verifying SSH connectivity to GitHub…"

for node_spec in "${NODES_ARRAY[@]}"; do
  node_dest="${node_spec%%:*}"
  node_label="${node_spec#*:}"
  if verify_github_ssh "$node_dest"; then
    ok "Node $node_label: SSH to GitHub works"
  else
    warn "Node $node_label: SSH to GitHub failed (Deploy Keys may not be registered yet)"
    VERIFY_FAILURES=$((VERIFY_FAILURES+1))
  fi
done

# ============================================================================
# Display public keys for manual Deploy Key registration
# ============================================================================

cat <<EOF

==> GitHub Deploy Key Registration
Visit: https://github.com/anonymuse/qw3/settings/keys/new

For each key below:
  1. Copy the full key (from 'ssh-ed25519' to the end)
  2. Paste in GitHub → Settings → Deploy keys → Add deploy key
  3. Title: Use the node label (e.g., "ds5-pro-1")
  4. Check "Allow write access"
  5. Click "Add key"

────────────────────────────────────────────────────────────────────────────────
EOF

for node_label in "${!PUBKEYS[@]}"; do
  echo ""
  echo "Node $node_label:"
  echo "${PUBKEYS[$node_label]}"
  echo ""
done

cat <<EOF
────────────────────────────────────────────────────────────────────────────────

After adding all 3 keys to GitHub, run the verification:
  bash ~/Code/qw3/tools/cluster/setup-github-auth.sh --verify-only

To test a push:
  ssh jesse@pro-1.local
  cd ~/Code/qw3
  echo "test" > .github-auth-verified
  git add .github-auth-verified
  git commit -m "test: GitHub auth verified"
  git push origin main
EOF

if [ "$VERIFY_FAILURES" -eq 0 ]; then
  say "All verifications passed! Deploy Keys are likely already registered."
else
  if [ "$VERIFY_ONLY" = 1 ]; then
    warn "$VERIFY_FAILURES verification(s) failed. Check the troubleshooting section in GITHUB-AUTH.md."
    exit 1
  else
    say "Setup complete. Verify by adding the Deploy Keys above to GitHub, then re-run with --verify-only"
  fi
fi
