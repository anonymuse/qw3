#!/usr/bin/env bash
# DS5 cluster node bootstrap — brand-new Apple Silicon Mac → agent-ready.
#
# Run ONCE per machine, in Terminal, as the logged-in admin user:
#   caffeinate -s bash bootstrap.sh --role primary      # the M5 Pro (orchestrator)
#   caffeinate -s bash bootstrap.sh --role worker        # each M5 Max
#   caffeinate -s bash bootstrap.sh --role worker --download   # the M5 Max that stores the model
#
# Idempotent: safe to re-run. It installs toolchain + Claude Code, enables SSH,
# generates an SSH key, and prints the node facts the orchestrator needs.
#
# It intentionally STOPS before authenticating Claude Code and before exchanging
# SSH keys across machines — those are the human/coordination seams (see README.md).

set -uo pipefail

ROLE="worker"
IS_DOWNLOAD=0
SHARE=0
for arg in "$@"; do
  case "$arg" in
    --role) : ;;                       # value handled below
    primary|worker) ROLE="$arg" ;;
    --download) IS_DOWNLOAD=1 ;;
    --share) SHARE=1 ;;
  esac
done
# allow "--role primary" form
prev=""
for arg in "$@"; do
  [ "$prev" = "--role" ] && ROLE="$arg"
  prev="$arg"
done

ZIG_WANT="0.16"
MIN_DISK_GB=150
say()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }

say "DS5 bootstrap — role=$ROLE download-node=$IS_DOWNLOAD"

# 0a. Must NOT run as root ---------------------------------------------------
# `sudo bash bootstrap.sh` breaks things: Homebrew hard-aborts as root, and the
# SSH key / Claude Code login must belong to your user, not root.
if [ "$(id -u)" -eq 0 ]; then
  warn "Don't run this with sudo/as root."
  cat <<EOF
        Homebrew refuses to install as root, and your SSH key + Claude Code login
        must belong to your user. Re-run as yourself, WITHOUT sudo:
            bash bootstrap.sh --role $ROLE $([ "$IS_DOWNLOAD" = 1 ] && echo --download)
        The script prompts for your password once when it actually needs sudo.
EOF
  exit 1
fi

# 0b. Require an Administrator account ---------------------------------------
if ! id -Gn "$USER" 2>/dev/null | grep -qw admin; then
  warn "User '$USER' is a Standard account, but this setup needs Administrator rights."
  cat <<EOF
        Homebrew and the sudo/SSH steps cannot run without admin. To fix, from an
        EXISTING administrator account on this Mac:
            sudo dseditgroup -o edit -a $USER -t user admin
        or: System Settings -> Users & Groups -> (i) next to $USER ->
            "Allow this user to administer this computer".
        Then log '$USER' out and back in, and re-run this script.
EOF
  exit 1
fi
ok "$USER is an Administrator"

# 0c. Prime sudo ONCE, up front ----------------------------------------------
# The Homebrew installer runs non-interactively (it won't prompt on its own) and
# the CLT install needs root too. Without a cached sudo credential Homebrew aborts
# with a misleading "needs to be an Administrator" message even for admins. Prompt
# for the password here, then keep the timestamp warm through the long installs.
say "Priming sudo — enter your login password once…"
if ! sudo -v; then
  warn "sudo failed. This account must be able to run sudo. Aborting."
  exit 1
fi
( while true; do sudo -n true 2>/dev/null; sleep 50; kill -0 "$$" 2>/dev/null || exit; done ) &
SUDO_KEEPALIVE=$!
trap '[ -n "${SUDO_KEEPALIVE:-}" ] && kill "$SUDO_KEEPALIVE" 2>/dev/null' EXIT
ok "sudo primed"

# 1. Xcode Command Line Tools ------------------------------------------------
if ! xcode-select -p >/dev/null 2>&1; then
  say "Installing Xcode Command Line Tools (headless)…"
  touch /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
  PROD=$(softwareupdate -l 2>/dev/null | grep -Eo 'Label: Command Line Tools.*' | sed 's/^Label: //' | tail -1)
  if [ -n "$PROD" ]; then
    sudo softwareupdate -i "$PROD" --verbose || warn "headless CLT install failed"
  fi
  rm -f /tmp/.com.apple.dt.CommandLineTools.installondemand.in-progress
  if ! xcode-select -p >/dev/null 2>&1; then
    warn "CLT still missing. Run 'xcode-select --install' and click through the GUI dialog, then re-run this script."
    exit 1
  fi
fi
ok "Xcode Command Line Tools present"

# 2. Homebrew ----------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  say "Installing Homebrew…"
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" \
    || { warn "Homebrew install failed"; exit 1; }
fi
eval "$(/opt/homebrew/bin/brew shellenv)"
grep -q 'brew shellenv' ~/.zprofile 2>/dev/null || \
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
ok "Homebrew present: $(brew --version | head -1)"

# 3. Toolchain: git, zig, node (for Claude Code), huggingface CLI ------------
say "Installing git, node, huggingface-cli…"
brew install git node >/dev/null 2>&1 || warn "brew install git/node had issues"
brew install huggingface-cli >/dev/null 2>&1 || warn "brew huggingface-cli unavailable; will try pip"
if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  python3 -m pip install --user -U huggingface_hub >/dev/null 2>&1 || warn "pip huggingface_hub failed"
fi

# Zig — the project pins 0.16, which may be a nightly/master build that brew's
# stable formula does NOT provide. Install, then verify hard.
say "Installing Zig (want $ZIG_WANT)…"
brew install zig >/dev/null 2>&1 || warn "brew install zig had issues"
ZIG_HAVE="$(zig version 2>/dev/null || echo none)"
case "$ZIG_HAVE" in
  ${ZIG_WANT}*) ok "Zig $ZIG_HAVE" ;;
  *) warn "Zig is '$ZIG_HAVE' but the project needs $ZIG_WANT.x.
        Download the matching build from https://ziglang.org/download/ (or the
        0.16 nightly), unpack to /opt/zig, and add it to PATH ahead of brew's.
        The orchestrator agent can also do this step for you." ;;
esac

# 4. Claude Code -------------------------------------------------------------
if ! command -v claude >/dev/null 2>&1; then
  say "Installing Claude Code (npm global)…"
  npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 \
    || warn "npm install of Claude Code failed — see https://claude.com/claude-code for the native installer"
fi
command -v claude >/dev/null 2>&1 && ok "Claude Code present: $(claude --version 2>/dev/null || echo installed)"

# 5. Enable Remote Login (SSH) ----------------------------------------------
say "Enabling Remote Login (SSH)…"
if sudo systemsetup -setremotelogin on 2>/dev/null; then
  ok "SSH enabled"
else
  warn "Could not toggle SSH from CLI (needs Full Disk Access for Terminal).
        Enable manually: System Settings → General → Sharing → Remote Login = ON."
fi

# 6. SSH key for the node mesh ----------------------------------------------
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
  say "Generating ed25519 SSH key…"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  ssh-keygen -t ed25519 -N "" -C "ds5-$(scutil --get LocalHostName 2>/dev/null || hostname)" -f "$HOME/.ssh/id_ed25519" >/dev/null
fi
ok "SSH public key ready"

# 7. Download-node disk check ------------------------------------------------
if [ "$IS_DOWNLOAD" = "1" ]; then
  AVAIL_GB=$(df -g "$HOME" | awk 'NR==2{print $4}')
  if [ "${AVAIL_GB:-0}" -lt "$MIN_DISK_GB" ]; then
    warn "Download node has ${AVAIL_GB}GB free; need >=${MIN_DISK_GB}GB for the model set."
  else
    ok "Download node free space: ${AVAIL_GB}GB"
  fi
  mkdir -p "$HOME/ds5-models"
fi

# 7b. Clone the DS5 repository ----------------------------------------------
CODE_DIR="$HOME/Code"
mkdir -p "$CODE_DIR"
REPO_DIR="${DS5_REPO_DIR:-$CODE_DIR/qw3}"
CLUSTER_DIR="$CODE_DIR/ds5-cluster"
mkdir -p "$CLUSTER_DIR"
REPO_URL="https://github.com/anonymuse/qw3.git"
if [ -d "$REPO_DIR/.git" ]; then
  say "Updating existing DS5 repo at $REPO_DIR…"
  git -C "$REPO_DIR" pull --ff-only || warn "git pull failed; leaving existing checkout as-is"
else
  say "Cloning DS5 repo to $REPO_DIR…"
  git clone "$REPO_URL" "$REPO_DIR" || warn "git clone failed — clone it manually: git clone $REPO_URL $REPO_DIR"
fi
[ -d "$REPO_DIR/.git" ] && ok "Repo ready at $REPO_DIR"

# 8. Node facts for the orchestrator ----------------------------------------
LOCALHOST="$(scutil --get LocalHostName 2>/dev/null || hostname)"
LANIP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo unknown)"
FACTS_FILE="$CLUSTER_DIR/node-facts-${LOCALHOST}.txt"
say "NODE FACTS — paste these to the orchestrator session"
cat > "$FACTS_FILE" <<EOF
  role:        $ROLE
  download:    $([ "$IS_DOWNLOAD" = 1 ] && echo yes || echo no)
  hostname:    ${LOCALHOST}.local
  lan_ip:      $LANIP
  user:        $USER
  zig:         $ZIG_HAVE
  claude:      $(command -v claude >/dev/null 2>&1 && echo installed || echo MISSING)
  repo:        $REPO_DIR $([ -d "$REPO_DIR/.git" ] && echo "(ready)" || echo "(MISSING)")
  ssh_pubkey:  $(cat "$HOME/.ssh/id_ed25519.pub")
EOF
cat "$FACTS_FILE"
say "Saved to $FACTS_FILE — AirDrop this file to Node A, or copy the block above."

if [ "${SHARE:-0}" = "1" ]; then
  say "Uploading NODE FACTS to a public paste service (termbin.com)…"
  URL=$(nc termbin.com 9999 < "$FACTS_FILE" 2>/dev/null | tr -d '\0')
  if [ -n "$URL" ]; then
    ok "Shared: $URL"
    warn "That URL is PUBLIC — it contains this node's hostname, LAN IP, and SSH PUBLIC key (no secrets, but anyone with the link can read it)."
  else
    warn "Share failed; use the saved file instead: $FACTS_FILE (AirDrop it)."
  fi
fi

say "NEXT (human): 1) run 'claude' once and finish the browser login.
              2) send the NODE FACTS block above to the orchestrator chat.
See tools/cluster/README.md for the full sequence."
